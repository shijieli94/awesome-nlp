import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.dataclass import ChoiceEnum
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.nonautoregressive_transformer import (
    NATransformerConfig,
    NATransformerModel,
    base_architecture,
)
from fairseq.models.transformer import Linear
from fairseq.modules.positional_embedding import PositionalEmbedding
from omegaconf import II
from torch import Tensor, nn

DecodeStrategyChoices = ChoiceEnum(["greedy", "lookahead", "viterbi", "full_viterbi", "sample", "beamsearch"])
TransitStrategyChoices = ChoiceEnum(["feature", "blank"])

logger = logging.getLogger(__name__)

FULL_PRECISION = False


def logsumexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    # Solving nan issue when x contains -inf
    # See https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim, keepdim=True)
    mask = m == float("-inf")
    m = m.detach()
    s = (x - m.masked_fill(mask, 0)).exp().sum(dim=dim, keepdim=True)
    s = s.masked_fill(mask, 1).log() + m.masked_fill(mask, float("-inf"))
    return s if keepdim else s.squeeze(dim)


def get_best_alignment(emission_lprobs, transit_lprobs, prev_output_mask, target_mask):
    bsz, pre_len, tgt_len = emission_lprobs.size()

    with torch.enable_grad():
        emission_lprobs.requires_grad_()

        cumulative_lprobs = emission_lprobs.new_full((bsz, pre_len), float("-inf"))
        cumulative_lprobs[:, 0] = emission_lprobs[:, 0, 0]

        for t in range(1, tgt_len):
            # [bsz, pre_len, 1] + [bsz, pre_len, pre_len]
            lprobs_t = cumulative_lprobs[:, :, None] + transit_lprobs
            lprobs_t = lprobs_t.max(dim=1)[0]
            lprobs_t += emission_lprobs[:, :, t]  # [bsz, pre_len]

            # if to the current position is invalid, we keep previous state
            cumulative_lprobs = torch.where(target_mask[:, [t]], lprobs_t, cumulative_lprobs)

        pre_eos_idx = prev_output_mask.sum(-1, keepdim=True) - 1
        cumulative_lprobs = cumulative_lprobs.gather(1, pre_eos_idx).squeeze(1)

        match_grad = torch.autograd.grad(cumulative_lprobs.sum(), [emission_lprobs])[0]

    best_alignment = match_grad.max(dim=1)[1]

    return best_alignment


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, d_output, dropout=0.0):
        super().__init__()
        self.input_to_hidden = nn.Linear(d_model, d_hidden)
        self.hidden_to_output = nn.Linear(d_hidden, d_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        h = F.relu(self.input_to_hidden(inputs))
        h = self.dropout(h)
        return self.hidden_to_output(h)


@dataclass
class DACRFTransformerModelConfig(NATransformerConfig):
    # upscale configs
    upsample_scale: float = field(
        default=2, metadata={"help": "Specifies the upsample scale for the decoder input length in training. "}
    )
    upsample_target_length: bool = field(
        default=False, metadata={"help": "use target lengths for upscale, otherwise source lengths will be used"}
    )
    # decoding configs
    decode_strategy: DecodeStrategyChoices = field(default="viterbi", metadata={"help": "Decoding strategy to use."})
    lookahead_beta: float = field(
        default=1,
        metadata={
            "help": "Parameter used to scale the score of logits in beamsearch decoding. "
            "The score of a sentence is given by: sum P(y_i|a_i) + beta * sum log(a_i|a_{i-1})"
        },
    )
    # glancing configs
    glance_p: Optional[str] = field(
        default=None,
        metadata={
            "help": "Set the glancing probability and its annealing schedule. "
            "For example, '0.5@0-0.1@200k' indicates annealing probability from 0.5 to 0.1 in the 0-200k steps."
        },
    )
    dag_loss_factor: float = field(default=1.0, metadata={"help": "Factor for the DAG loss"})
    # CRF configs
    fix_dag_params: bool = field(
        default=False, metadata={"help": "Whether to fix the DAG parameters during CRF training."}
    )
    dacrf_loss_factor: float = field(default=0.0, metadata={"help": "Factor for the DAG loss in the CRF"})
    crf_lowrank_approx: int = field(default=32, metadata={"help": "Specifies the dimension of transition embeddings."})
    crf_beam_approx: int = field(default=64, metadata={"help": "Beam size for apporixmating the normalizing factor"})
    crf_decode_beam: int = field(default=II("model.crf_beam_approx"), metadata={"help": "Beam size for decoding"})
    dynamic_crf: bool = field(default=False, metadata={"help": "Whether to use dynamic CRF."})


@register_model("dacrf_transformer", dataclass=DACRFTransformerModelConfig)
class DACRFTransformerModel(NATransformerModel):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.glat_scheduler = utils.get_anneal_argument_parser(cfg.glance_p) if cfg.glance_p is not None else None

        self.link_positional = PositionalEmbedding(
            self.decoder.max_positions(), cfg.decoder.embed_dim, self.pad, learned=True
        )
        self.query_linear = Linear(cfg.decoder.embed_dim * 2, cfg.decoder.embed_dim, bias=False)
        self.key_linear = Linear(cfg.decoder.embed_dim * 2, cfg.decoder.embed_dim, bias=False)
        self.gate_linear = Linear(cfg.decoder.embed_dim * 2, cfg.decoder.attention_heads, bias=False)

        if cfg.fix_dag_params:
            for param in self.parameters():
                param.requires_grad = False

        if cfg.dacrf_loss_factor > 0:
            self.embed_query = nn.Embedding(len(self.tgt_dict), cfg.crf_lowrank_approx, padding_idx=self.pad)
            self.embed_key = nn.Embedding(len(self.tgt_dict), cfg.crf_lowrank_approx, padding_idx=self.pad)
            self.dynamic_crf = getattr(cfg, "dynamic_crf", False)
            if self.dynamic_crf:
                self.crf_ffn = FeedForward(
                    cfg.decoder.embed_dim * 2, cfg.decoder.embed_dim, cfg.crf_lowrank_approx**2, dropout=cfg.dropout
                )

    @property
    def allow_ensemble(self):
        return False

    def _initialize_output_tokens(self, encoder_out, src_tokens, lengths):
        lengths = (lengths * self.cfg.upsample_scale).long()
        decoder_out = self._initialize_output_tokens_with_length(encoder_out, src_tokens, lengths)
        return decoder_out.output_tokens

    def right_triangular_mask(self, prev_output_mask):
        # return a mask matrix whose elements only allow transition from left to right
        bsz, max_len = prev_output_mask.size()

        if hasattr(self, "_right_triangular_mask") and self._right_triangular_mask.size(-1) >= max_len:
            _right_triangular_mask = self._right_triangular_mask[:max_len, :max_len].expand(bsz, -1, -1)
            return _right_triangular_mask.to(prev_output_mask)

        self._right_triangular_mask = torch.ones(max_len, max_len).triu_(1)
        return self.right_triangular_mask(prev_output_mask)

    def compute_transit_lprobs(self, features, prev_output_tokens):
        bsz, pre_len, _ = features.size()

        num_heads = self.cfg.decoder.attention_heads
        head_dim = self.cfg.decoder.embed_dim // self.cfg.decoder.attention_heads

        prev_output_mask = prev_output_tokens.ne(self.pad).unsqueeze(1).repeat(1, num_heads, 1).view(-1, pre_len)
        valid_transit_mask = prev_output_mask.unsqueeze(1) & self.right_triangular_mask(prev_output_mask)

        # [pre_len, bsz, dim]
        features = torch.cat([features, self.link_positional(prev_output_tokens)], -1).transpose(0, 1)

        # Use multiple heads in calculating transition matrix
        q_chunks = self.query_linear(features).contiguous().view(pre_len, bsz * num_heads, -1).transpose(0, 1)
        k_chunks = self.key_linear(features).contiguous().view(pre_len, bsz * num_heads, -1).transpose(0, 1)
        gates = self.gate_linear(features).transpose(0, 1)

        if FULL_PRECISION:
            q_chunks, k_chunks, gates = q_chunks.float(), k_chunks.float(), gates.float()

        # Transition probability for each head, with shape batch_size * pre_len * pre_len * chunk_num
        transit_scores = torch.bmm(q_chunks, k_chunks.transpose(1, 2)) / head_dim**0.5
        transit_scores = transit_scores.masked_fill(~valid_transit_mask, float("-inf"))

        transit_lprobs = F.log_softmax(transit_scores, dim=-1)
        transit_lprobs = transit_lprobs.masked_fill(~valid_transit_mask, float("-inf"))
        transit_lprobs = transit_lprobs.view(bsz, num_heads, pre_len, pre_len).permute(0, 2, 3, 1)

        log_gates = F.log_softmax(gates, dim=-1)

        transit_lprobs = logsumexp(transit_lprobs + log_gates.unsqueeze(2), dim=-1)

        return transit_lprobs

    def extract_features(self, encoder_out, prev_output_tokens, **kwargs):
        features, _ = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, embedding_copy=False)
        word_ins_out = self.output_layer(features)
        return word_ins_out, features

    @torch.no_grad()
    def glancing_sampling(self, encoder_out, prev_output_tokens, target_tokens, glat_ratio):
        bsz, pre_len = prev_output_tokens.size()

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)
        nonspecial_mask = prev_output_mask & prev_output_tokens.ne(self.bos) & prev_output_tokens.ne(self.eos)

        with utils.model_eval(self.decoder):
            emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        if FULL_PRECISION:
            emission_scores = emission_scores.float()

        transit_lprobs = self.compute_transit_lprobs(features, prev_output_tokens)

        oracle_predictions = emission_scores.max(dim=-1)[1]
        oracle_predictions = torch.where(nonspecial_mask, oracle_predictions, prev_output_tokens)

        emission_lprobs = F.log_softmax(emission_scores, dim=-1)
        emission_lprobs = emission_lprobs.gather(2, target_tokens[:, None, :].expand(-1, pre_len, -1))

        best_alignment = get_best_alignment(emission_lprobs, transit_lprobs, prev_output_mask, target_mask)

        scattered_predictions = oracle_predictions.scatter(1, best_alignment, target_tokens)
        scattered_predictions = torch.where(nonspecial_mask, scattered_predictions, prev_output_tokens)

        scattered_mask = torch.zeros_like(prev_output_tokens).scatter(1, best_alignment, 1).bool()
        scattered_mask &= nonspecial_mask

        unmatched_num = (oracle_predictions != scattered_predictions).sum(1, keepdim=True)  # noqa

        probs = torch.zeros_like(oracle_predictions).float().uniform_().masked_fill_(~scattered_mask, -2)
        probs_thresh = probs.sort(descending=True)[0].gather(-1, (unmatched_num * glat_ratio + 0.5).long())
        # only positions whose probs are higher than the threshold will be replaced by the prediction
        keep_mask = probs <= probs_thresh

        glat_prev_output_tokens = scattered_predictions.clone()
        glat_prev_output_tokens[keep_mask] = prev_output_tokens[keep_mask]

        total = (target_mask.sum(-1) - 2).sum()
        n_correct = total - unmatched_num.sum()
        glat_info = {"_glat@total": utils.item(total), "_glat@n_correct": utils.item(n_correct)}

        return glat_prev_output_tokens, glat_info, best_alignment, scattered_predictions

    @torch.no_grad()
    def get_best_alignment(self, encoder_out, prev_output_tokens, target_tokens):
        bsz, pre_len = prev_output_tokens.size()

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)

        emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        if FULL_PRECISION:
            emission_scores = emission_scores.float()

        transit_lprobs = self.compute_transit_lprobs(features, prev_output_tokens)

        emission_lprobs = F.log_softmax(emission_scores, dim=-1)
        emission_lprobs = emission_lprobs.gather(2, target_tokens[:, None, :].expand(-1, pre_len, -1))

        best_alignment = get_best_alignment(emission_lprobs, transit_lprobs, prev_output_mask, target_mask)

        return best_alignment

    def _compute_dacrf_numerator(self, emissions, features, targets, masks=None):
        lowrank = self.cfg.crf_lowrank_approx

        emission_scores = emissions.gather(2, targets[:, :, None])[:, :, 0]  # B x T

        E1 = self.embed_query(targets[:, :-1]).view(-1, lowrank)
        E2 = self.embed_key(targets[:, 1:]).view(-1, lowrank)

        if self.dynamic_crf:
            D = self.crf_ffn(torch.cat([features[:, :-1], features[:, 1:]], dim=-1)).view(-1, lowrank, lowrank)
            transition_scores = torch.einsum("bt,btd->bd", E1, D)
            transition_scores = torch.einsum("bd, bd->b", transition_scores, E2).view(emissions.size(0), -1)
        else:
            transition_scores = (E1 * E2).sum(1)

        transition_scores = transition_scores.view(emission_scores.size(0), -1)
        scores = emission_scores
        scores[:, 1:] += transition_scores

        scores = scores * masks.type_as(scores)

        return scores.sum(-1)

    def _compute_dacrf_normalizer(self, emissions, features, targets=None, masks=None):
        beam = self.cfg.crf_beam_approx
        lowrank = self.cfg.crf_lowrank_approx

        bsz, seq_len = emissions.shape[:2]

        _emissions = emissions.scatter(2, targets[:, :, None], float("inf"))
        beam_targets = _emissions.topk(beam, 2)[1]
        beam_emission_scores = emissions.gather(2, beam_targets)

        E1 = self.embed_query(beam_targets[:, :-1]).view(-1, beam, lowrank)
        E2 = self.embed_key(beam_targets[:, 1:]).view(-1, beam, lowrank)

        if self.dynamic_crf:
            D = self.crf_ffn(torch.cat([features[:, :-1], features[:, 1:]], dim=-1)).view(-1, lowrank, lowrank)
            beam_transition_matrix = torch.einsum("bkt,btd->bkd", E1, D)
            beam_transition_matrix = torch.einsum("bkd,bld->bkl", beam_transition_matrix, E2)
        else:
            beam_transition_matrix = torch.bmm(E1, E2.transpose(1, 2))

        beam_transition_matrix = beam_transition_matrix.view(bsz, -1, beam, beam)

        # compute the normalizer in the log-space
        score = beam_emission_scores[:, 0]  # B x K
        for i in range(1, seq_len):
            next_score = score[:, :, None] + beam_transition_matrix[:, i - 1]
            next_score = logsumexp(next_score, dim=1) + beam_emission_scores[:, i]

            score = torch.where(masks[:, i : i + 1], next_score, score)

        # Sum (log-sum-exp) over all possible tags
        return logsumexp(score, dim=1)

    def compute_dag_loss(self, prev_output_tokens, target_tokens, *, emission_lprobs, transit_lprobs):
        bsz, pre_len, tgt_len = emission_lprobs.size()

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)

        cumulative_lprobs = emission_lprobs.new_full((bsz, pre_len), float("-inf"))
        cumulative_lprobs[:, 0] = emission_lprobs[:, 0, 0]

        for t in range(1, tgt_len):
            # [bsz, pre_len, 1] + [bsz, pre_len, pre_len]
            lprobs_t = cumulative_lprobs[:, :, None] + transit_lprobs
            lprobs_t = logsumexp(lprobs_t, dim=1)
            lprobs_t += emission_lprobs[:, :, t]  # [bsz, 1, pre_len]

            # only compute the ground-truth path
            cumulative_lprobs = torch.where(target_mask[:, [t]], lprobs_t, cumulative_lprobs)

        pre_eos_idx = prev_output_mask.sum(-1, keepdim=True) - 1
        dacrf_loss = -cumulative_lprobs.gather(1, pre_eos_idx).squeeze(1)

        invalid_masks = dacrf_loss.isnan()
        if invalid_masks.sum() > 0:
            logger.warning(f"{invalid_masks.sum()} samples have nan da_loss.")
            dacrf_loss = dacrf_loss.masked_fill(invalid_masks, 0)

        invalid_masks = dacrf_loss.isinf()
        if invalid_masks.sum() > 0:
            logger.warning(f"{invalid_masks.sum()} samples have inf loss. Please use a larger upsample value")
            dacrf_loss = dacrf_loss.masked_fill(invalid_masks, 0)

        return (dacrf_loss / target_mask.type_as(dacrf_loss).sum(-1)).mean()

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        _ret = {"output_logs": {}}
        if self.cfg.length_loss_factor > 0:
            # length prediction
            length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
            length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
            _ret["length"] = {"out": length_out, "tgt": length_tgt, "factor": self.cfg.length_loss_factor}

        # in this case, input length was determinate by the upsample scale
        # to avoid adding too many padding tokens, we use the same scale for all sentences
        lengths = tgt_tokens.ne(self.pad).sum(1).long() if self.cfg.upsample_target_length else src_lengths
        prev_output_tokens = self._initialize_output_tokens(encoder_out, src_tokens, lengths)

        glat_ratio = None if self.glat_scheduler is None else max(0, self.glat_scheduler(self.get_num_updates()))
        if glat_ratio is not None and glat_ratio > 0:
            prev_output_tokens, glat_info, best_alignment, _ = self.glancing_sampling(
                encoder_out, prev_output_tokens, tgt_tokens, glat_ratio
            )
            _ret["output_logs"].update(glat_info)

        if glat_ratio is None:
            best_alignment = self.get_best_alignment(encoder_out, prev_output_tokens, tgt_tokens)

        # decoding
        emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        if FULL_PRECISION:
            emission_scores = emission_scores.float()

        transit_lprobs = self.compute_transit_lprobs(features, prev_output_tokens)

        emission_lprobs = F.log_softmax(emission_scores, dim=-1)
        emission_lprobs = emission_lprobs.gather(2, tgt_tokens[:, None, :].expand(-1, prev_output_tokens.size(1), -1))

        ret = {}
        if self.cfg.dag_loss_factor > 0:
            dag_loss = self.compute_dag_loss(
                prev_output_tokens, tgt_tokens, emission_lprobs=emission_lprobs, transit_lprobs=transit_lprobs
            )
            dag_loss *= self.cfg.dag_loss_factor
            ret["dag_loss"] = {"loss": dag_loss, "factor": self.cfg.dag_loss_factor}

        if self.cfg.dacrf_loss_factor > 0:
            prev_output_masks = prev_output_tokens.ne(self.pad)
            target_masks = tgt_tokens.ne(self.pad)

            _emission_scores = emission_scores.gather(
                1, best_alignment[:, :, None].expand(-1, -1, emission_scores.size(-1))
            )
            _features = features.gather(1, best_alignment[:, :, None].expand(-1, -1, features.size(-1)))
            numerator = self._compute_dacrf_numerator(_emission_scores, _features, tgt_tokens, target_masks)
            denominator = self._compute_dacrf_normalizer(_emission_scores, _features, tgt_tokens, target_masks)

            # if the below condition is not met, the best_alignment would contain errors.
            valid_masks = prev_output_masks.sum(-1) >= target_masks.sum(-1)
            dacrf_loss = -(numerator[valid_masks] - denominator[valid_masks])
            dacrf_loss = dacrf_loss / target_masks[valid_masks].type_as(dacrf_loss).sum(-1)
            dacrf_loss = dacrf_loss.masked_fill(dacrf_loss <= 0, 0)

            dacrf_loss = dacrf_loss.mean() * self.cfg.dacrf_loss_factor

            ret["dacrf_loss"] = {"loss": dacrf_loss, "factor": self.cfg.dacrf_loss_factor}

        ret.update(**_ret)

        return ret

    def forward_decoder(self, decoder_out, encoder_out, src_tokens=None, decoding_format=None, **kwargs):
        history = decoder_out.history

        # output_tokens have the predicted length while prev_output_tokens have the up-scaled length
        _tokens = decoder_out.output_tokens if self.cfg.upsample_target_length else src_tokens
        lengths = _tokens.ne(self.pad).sum(1).long()

        output_tokens = self._initialize_output_tokens(encoder_out, src_tokens, lengths)

        # decoding
        if self.cfg.decode_strategy in ["lookahead", "greedy"]:
            output_tokens, output_scores = self._inference_lookahead(
                encoder_out, output_tokens, decoder_out.output_tokens
            )
        elif self.cfg.decode_strategy == "viterbi":
            output_tokens, output_scores = self._inference_viterbi(
                encoder_out, output_tokens, decoder_out.output_tokens
            )
        elif self.cfg.decode_strategy == "full_viterbi":
            output_tokens, output_scores = self._inference_full_viterbi(
                encoder_out, output_tokens, decoder_out.output_tokens
            )

        if history is not None:
            history.append({"tokens": output_tokens.clone(), "scores": output_scores.clone()})  # noqa

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def _inference_lookahead(self, encoder_out, prev_output_tokens, target_tokens):
        prev_output_mask = prev_output_tokens.ne(self.pad)

        emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        if FULL_PRECISION:
            emission_scores = emission_scores.float()
        emission_lprobs = F.log_softmax(emission_scores, dim=-1)

        transit_lprobs = self.compute_transit_lprobs(features, prev_output_tokens)

        unreduced_lprobs, unreduced_tokens = emission_lprobs.max(dim=-1)

        unreduced_tokens = unreduced_tokens.tolist()
        output_length = prev_output_mask.sum(-1).tolist()

        if self.cfg.decode_strategy == "lookahead":
            links_idx = (
                (transit_lprobs + unreduced_lprobs.unsqueeze(1) * self.cfg.lookahead_beta).max(dim=-1)[1].tolist()
            )
        elif self.cfg.decode_strategy == "greedy":
            links_idx = transit_lprobs.max(dim=-1)[1].tolist()

        unpad_output_tokens = []
        for i, length in enumerate(output_length):
            last = unreduced_tokens[i][0]
            j = 0
            res = [last]
            while j != length - 1:
                j = links_idx[i][j]  # noqa
                now_token = unreduced_tokens[i][j]
                if now_token != self.pad and now_token != last:
                    res.append(now_token)
                last = now_token
            unpad_output_tokens.append(res)

        output_seqlen = max([len(res) for res in unpad_output_tokens])
        output_tokens = [res + [self.pad] * (output_seqlen - len(res)) for res in unpad_output_tokens]
        output_tokens = prev_output_tokens.new_tensor(output_tokens)
        output_scores = torch.full_like(output_tokens, 1.0, dtype=torch.float)
        return output_tokens, output_scores

    def _inference_viterbi(self, encoder_out, prev_output_tokens, target_tokens):
        bsz, tgt_len = target_tokens.size()
        pre_len = prev_output_tokens.size(1)

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)

        invalid_batch_mask = (prev_output_mask.sum(-1) < target_mask.sum(-1)).bool()  # noqa
        if invalid_batch_mask.any():
            if tgt_len < pre_len:
                target_mask[invalid_batch_mask] = prev_output_mask[invalid_batch_mask][:, :tgt_len]
            else:
                target_mask[invalid_batch_mask] = False
                target_mask[invalid_batch_mask][:, :pre_len] = prev_output_mask[invalid_batch_mask]

        emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        if FULL_PRECISION:
            emission_scores = emission_scores.float()
        emission_lprobs = F.log_softmax(emission_scores, dim=-1)

        output_scores, output_tokens = emission_lprobs.max(dim=-1)

        transit_lprobs = self.compute_transit_lprobs(features, prev_output_tokens)

        cumulative_lprobs = torch.full_like(output_scores, float("-inf"))
        # at the first step we only consider the emission score of the first position
        cumulative_lprobs[:, 0] = output_scores[:, 0]

        pre_eos_idx = prev_output_mask.sum(-1, keepdim=True) - 1

        traj_index = []
        for t in range(1, tgt_len):
            # [bsz, pre_len, 1] + [bsz, pre_len, pre_len]
            lprobs_t = cumulative_lprobs[:, :, None] + transit_lprobs
            cumulative_lprobs, index_t = lprobs_t.max(dim=1)
            cumulative_lprobs += output_scores  # [bsz, 1, pre_len]

            # if to the current position is invalid, we set the pointer index of all tokens to eos position
            index_t = torch.where(target_mask[:, [t]], index_t, pre_eos_idx)
            traj_index.append(index_t)

        # max_length * batch
        best_alignment = [pre_eos_idx]
        for index in reversed(traj_index):
            best_alignment.insert(0, index.gather(1, best_alignment[0]))
        best_alignment = torch.cat(best_alignment, 1)

        if self.cfg.dacrf_loss_factor > 0:
            _emission_scores = emission_scores.gather(
                1, best_alignment[:, :, None].expand(-1, -1, emission_scores.size(-1))
            )
            _features = features.gather(1, best_alignment[:, :, None].expand(-1, -1, features.size(-1)))
            output_tokens, output_scores = self._crf_viterbi_decode(_emission_scores, _features, masks=target_mask)
        else:
            output_scores = output_scores.gather(1, best_alignment)
            output_tokens = output_tokens.gather(1, best_alignment)

        # cleaned_output_tokens = torch.full_like(output_tokens, self.pad)
        # for i in range(output_tokens.size(0)):
        #     tokens = output_tokens[i].unique_consecutive()
        #     cleaned_output_tokens[i, : tokens.size(0)] = tokens
        #
        # return cleaned_output_tokens, torch.zeros_like(cleaned_output_tokens, dtype=output_scores.dtype)
        return output_tokens, output_scores

    def _crf_viterbi_decode(self, emissions, features, masks=None):
        beam = self.cfg.crf_decode_beam
        lowrank = self.cfg.crf_lowrank_approx

        batch_size, seq_len = emissions.shape[:2]
        beam_emission_scores, beam_targets = emissions.topk(beam, 2)

        E1 = self.embed_query(beam_targets[:, :-1]).view(-1, beam, lowrank)
        E2 = self.embed_key(beam_targets[:, 1:]).view(-1, beam, lowrank)
        if self.dynamic_crf:
            D = self.crf_ffn(torch.cat([features[:, :-1], features[:, 1:]], dim=-1)).view(-1, lowrank, lowrank)
            beam_transition_matrix = torch.einsum("bkt,btd->bkd", E1, D)
            beam_transition_matrix = torch.einsum("bkd,bld->bkl", beam_transition_matrix, E2)
        else:
            beam_transition_matrix = torch.bmm(E1, E2.transpose(1, 2))

        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam)

        traj_tokens, traj_scores = [], []
        finalized_tokens, finalized_scores = [], []

        # compute the normalizer in the log-space
        score = beam_emission_scores[:, 0]  # B x K
        dummy = utils.new_arange(score)

        for i in range(1, seq_len):
            traj_scores.append(score)
            _score = score[:, :, None] + beam_transition_matrix[:, i - 1]
            _score, _index = _score.max(dim=1)
            _score = _score + beam_emission_scores[:, i]

            if masks is not None:
                score = torch.where(masks[:, i : i + 1], _score, score)
                index = torch.where(masks[:, i : i + 1], _index, dummy)
            else:
                score, index = _score, _index
            traj_tokens.append(index)

        # now running the back-tracing and find the best
        best_score, best_index = score.max(dim=1)
        finalized_tokens.append(best_index[:, None])
        finalized_scores.append(best_score[:, None])

        for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)):
            previous_index = finalized_tokens[-1]
            finalized_tokens.append(idx.gather(1, previous_index))
            finalized_scores.append(scs.gather(1, previous_index))

        finalized_tokens.reverse()
        finalized_tokens = torch.cat(finalized_tokens, 1)
        finalized_tokens = beam_targets.gather(2, finalized_tokens[:, :, None])[:, :, 0]

        finalized_scores.reverse()
        finalized_scores = torch.cat(finalized_scores, 1)
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1]

        return finalized_tokens, finalized_scores

    def _inference_full_viterbi(self, encoder_out, prev_output_tokens, target_tokens):
        bsz, pre_len = prev_output_tokens.size()
        tgt_len = target_tokens.size(1)

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)

        invalid_batch_mask = (prev_output_mask.sum(-1) < target_mask.sum(-1)).bool()  # noqa
        if invalid_batch_mask.any():
            if tgt_len < pre_len:
                target_mask[invalid_batch_mask] = prev_output_mask[invalid_batch_mask][:, :tgt_len]
            else:
                target_mask[invalid_batch_mask] = False
                target_mask[invalid_batch_mask][:, :pre_len] = prev_output_mask[invalid_batch_mask]

        emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        if FULL_PRECISION:
            emission_scores = emission_scores.float()
        emission_lprobs = F.log_softmax(emission_scores, dim=-1)

        del emission_scores

        transit_lprobs = self.compute_transit_lprobs(features, prev_output_tokens)

        beam = self.cfg.crf_decode_beam
        beam_scores, beam_tokens = emission_lprobs.topk(beam, 2)

        crf_scores = torch.einsum("bsld,btmd->bstlm", self.embed_query(beam_tokens), self.embed_key(beam_tokens))
        crf_scores = crf_scores.contiguous().view(-1, beam * beam).log_softmax(-1).view_as(crf_scores)
        # combine CRF scores with beam scores to avoid multiple computing
        crf_scores = crf_scores + beam_scores[:, None, :, None, :]  # [bsz, 1, tgt_len, 1, beam]

        cumulative_scores = beam_scores

        # useful for passing pad positions
        pre_eos_idx = prev_output_mask.sum(-1, keepdim=True) - 1
        identity_map = utils.new_arange(cumulative_scores)

        traj_timestep_index, traj_beam_index = [], []
        for t in range(1, tgt_len):
            scores_t = (
                cumulative_scores[:, :, None, :, None]  # [bsz, pre_len, 1, beam, 1]
                + transit_lprobs[:, :, :, None, None]  # [bsz, pre_len, pre_len, 1, 1]
                + crf_scores
            )

            # get the best index over the timestep dimension
            scores_t, timestep_index = scores_t.max(dim=1)  # [bsz, pre_len, beam, beam]

            # if the current transit is invalid, we set the timestep to eos_idx
            timestep_index = torch.where(
                target_mask[:, [t]][:, :, None, None], timestep_index, pre_eos_idx[:, :, None, None]
            )
            traj_timestep_index.append(timestep_index)

            # get the best index over the beam dimension
            scores_t, beam_index = scores_t.max(dim=2)  # [bsz, pre_len, beam]

            beam_index = torch.where(target_mask[:, [t]][:, :, None], beam_index, identity_map)
            traj_beam_index.append(beam_index)

            cumulative_scores = torch.where(target_mask[:, [t]][:, :, None], scores_t, cumulative_scores)

        # start from the eos idx and eos beam, both with the shape of [bsz, 1]
        best_timestep = [pre_eos_idx]
        best_beam = [torch.full_like(pre_eos_idx, 0)]

        for beam_idx, timestep_idx in zip(reversed(traj_beam_index), reversed(traj_timestep_index)):
            # first get the best previous beam index for all positions in the current beam
            beam_idx = beam_idx.gather(1, best_timestep[0][:, :, None].expand(-1, -1, beam)).squeeze(1)  # [bsz, beam]
            # then get the best previous beam index for the best positions
            beam_idx = beam_idx.gather(1, best_beam[0])

            # first get the best previous timestep for all pairs in the current beam
            timestep_idx = timestep_idx.gather(
                1, best_timestep[0][:, :, None, None].expand(-1, -1, beam, beam)
            ).squeeze(1)
            timestep_idx = timestep_idx[torch.arange(bsz), beam_idx.squeeze(1), best_beam[0].squeeze(1)]

            best_beam.insert(0, beam_idx)
            best_timestep.insert(0, timestep_idx.unsqueeze(1))

        best_timestep = torch.cat(best_timestep, 1)
        best_beam = torch.cat(best_beam, 1)

        # first gather all beams in the selected path
        output_tokens = beam_tokens.gather(1, best_timestep[:, :, None].expand(-1, -1, beam))
        # then gather all tokens from the selected beams
        output_tokens = output_tokens.gather(2, best_beam[:, :, None]).squeeze(2)

        output_scores = emission_lprobs.new_full(output_tokens.size(), 1.0)

        return output_tokens, output_scores


@register_model_architecture("dacrf_transformer", "dacrf_transformer_wmt_en_de")
def dacrf_transformer_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("dacrf_transformer", "dacrf_transformer_iwslt_de_en")
def dacrf_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("dacrf_transformer", "dacrf_transformer_big")
def dacrf_transformer_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)
