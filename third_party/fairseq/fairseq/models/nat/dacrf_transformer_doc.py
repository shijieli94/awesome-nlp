import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.dataclass import ChoiceEnum
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.nonautoregressive_transformer import (
    NATransformerDecoder,
    NATransformerEncoder,
    base_architecture,
    ensemble_decoder,
    init_bert_params,
)
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayerBase,
    TransformerEncoderLayerBase,
)
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from .dacrf_transformer import (
    DACRFTransformerModel,
    DACRFTransformerModelConfig,
    Linear,
    get_best_alignment,
    logsumexp,
)

DecodeStrategyChoices = ChoiceEnum(["greedy", "lookahead", "viterbi", "joint-viterbi", "sample", "beamsearch"])
TransitStrategyChoices = ChoiceEnum(["feature", "blank"])

logger = logging.getLogger(__name__)

FULL_PRECISION = False


def get_segment_mask(segment_ids):
    segment_groups = utils.new_arange(segment_ids, segment_ids.max()) + 1
    segment_mask = segment_ids[:, None, :] == segment_groups[None, :, None]
    return segment_mask


def get_sent_lens(mask):
    sent_lens = mask.sum(-1).long().view(-1)
    sent_lens = sent_lens[sent_lens.ne(0)]
    return sent_lens


def expand_segment_dim(segment_mask, *tensors):
    chunk_size = segment_mask.size(1)
    new_tensors = []

    for tensor in tensors:
        expand_dims = (
            -1,
            chunk_size,
        ) + (
            -1,
        ) * (tensor.ndim - 1)
        tensor = tensor.unsqueeze(1).expand(*expand_dims)
        new_tensors.append(tensor[segment_mask])
    return new_tensors if len(new_tensors) > 1 else new_tensors[0]


def pad_to_new_tensor(tensor, lens, pad):
    tensor = torch.split(tensor, lens.tolist(), dim=0)
    tensor = pad_sequence(tensor, batch_first=True, padding_value=pad)
    return tensor


def scatter_to_tensor(new_tensor, old_tensor, segment_ids, sent_lens):
    segment_ids = segment_ids.amax(-1).cumsum(-1)

    bsz_idx, pos_idx = 0, 0
    for seg_idx, (lens, tensor) in enumerate(zip(sent_lens, old_tensor)):
        new_tensor[bsz_idx, pos_idx : pos_idx + lens] = tensor[:lens]
        pos_idx += lens

        if seg_idx == segment_ids[bsz_idx] - 1:
            bsz_idx += 1
            pos_idx = 0


@dataclass
class GroupDACRFTransformerConfig(DACRFTransformerModelConfig):
    encoder_ctxlayers: int = field(
        default=0, metadata={"help": "how many layers for global attention in encoder self-attention."}
    )
    no_encoder_local: bool = field(default=False, metadata={"help": "do not use local attention in encoder."})
    decoder_ctxlayers: int = field(
        default=0, metadata={"help": "how many layers for global attention in decoder self-attention."}
    )
    no_decoder_local: bool = field(default=False, metadata={"help": "do not use local attention in decoder."})
    cross_ctxlayers: int = field(
        default=0, metadata={"help": "how many layers for global attention in decoder cross-attention."}
    )
    no_cross_local: bool = field(
        default=False, metadata={"help": "do not use local attention in decoder cross-attention."}
    )


@register_model("dacrf_transformer_doc", dataclass=GroupDACRFTransformerConfig)
class GroupDACRFTransformerModel(DACRFTransformerModel):
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        encoder = GroupDACRFTransformerEncoder(cfg, src_dict, embed_tokens)
        if cfg.apply_bert_init:
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = GroupDACRFTransformerDecoder(cfg, tgt_dict, embed_tokens)
        if cfg.apply_bert_init:
            decoder.apply(init_bert_params)
        return decoder

    def _initialize_output_tokens(self, encoder_out, src_tokens, lengths):
        lengths = (lengths * self.cfg.upsample_scale).long()
        decoder_out = self._initialize_output_tokens_with_length(encoder_out, src_tokens, lengths)
        prev_output_tokens = decoder_out.output_tokens

        # scatter the bos and eos tokens into right positions
        src_segment_mask = get_segment_mask(encoder_out["segment_ids"][0])
        src_lengs = (src_segment_mask.sum(-1) * self.cfg.upsample_scale).long().cumsum(-1)
        bos_indices = src_lengs.masked_fill(src_lengs == src_lengs.amax(-1, keepdim=True), 0)

        prev_output_tokens = prev_output_tokens.scatter(1, bos_indices, self.bos)
        prev_output_tokens = prev_output_tokens.scatter(1, src_lengs - 1, self.eos)
        return prev_output_tokens

    def compute_transit_lprobs(self, features, prev_output_tokens):
        num_heads = self.cfg.decoder.attention_heads
        head_dim = self.cfg.decoder.embed_dim // self.cfg.decoder.attention_heads

        # [pre_len, bsz, dim]
        features = torch.cat([features, self.link_positional(prev_output_tokens)], -1)

        segment_ids = self.decoder.get_segment_ids(prev_output_tokens, padding_mask=prev_output_tokens.eq(self.pad))
        segment_mask = get_segment_mask(segment_ids)
        sent_lens = get_sent_lens(segment_mask)

        features, prev_output_tokens = expand_segment_dim(segment_mask, features, prev_output_tokens)

        assert features.size(0) == prev_output_tokens.size(0) == sent_lens.sum()

        features = pad_to_new_tensor(features, sent_lens, pad=0)
        prev_output_tokens = pad_to_new_tensor(prev_output_tokens, sent_lens, pad=self.pad)

        bsz, pre_len, _ = features.size()

        features = features.transpose(0, 1)

        prev_output_mask = prev_output_tokens.ne(self.pad).unsqueeze(1).repeat(1, num_heads, 1).view(-1, pre_len)
        valid_transit_mask = prev_output_mask.unsqueeze(1) & self.right_triangular_mask(prev_output_mask)

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
        source_segment_ids = encoder_out["segment_ids"][0]
        target_segment_ids = self.decoder.get_segment_ids(
            prev_output_tokens, padding_mask=prev_output_tokens.eq(self.pad)
        )

        features, _ = self.decoder.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=False,
            source_segment_ids=source_segment_ids,
            target_segment_ids=target_segment_ids,
        )
        word_ins_out = self.output_layer(features)

        return word_ins_out, features

    @torch.no_grad()
    def glancing_sampling(self, encoder_out, prev_output_tokens, target_tokens, glat_ratio):
        prev_output_tokens_clone = prev_output_tokens.clone()

        with utils.model_eval(self.decoder):
            emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        if FULL_PRECISION:
            emission_scores = emission_scores.float()

        transit_lprobs = self.compute_transit_lprobs(features, prev_output_tokens)

        # convert the prev_output_tokens and emission_scores into segment version
        prev_segment_ids = self.decoder.get_segment_ids(
            prev_output_tokens, padding_mask=prev_output_tokens.eq(self.pad)
        )
        prev_segment_mask = get_segment_mask(prev_segment_ids)
        prev_sent_lens = get_sent_lens(prev_segment_mask)

        emission_scores, prev_output_tokens = expand_segment_dim(
            prev_segment_mask, emission_scores, prev_output_tokens
        )

        assert prev_output_tokens.size(0) == emission_scores.size(0) == prev_sent_lens.sum()

        emission_scores = pad_to_new_tensor(emission_scores, prev_sent_lens, pad=0)
        prev_output_tokens = pad_to_new_tensor(prev_output_tokens, prev_sent_lens, pad=self.pad)

        # convert the target_tokens into segment version
        tgt_segment_ids = self.decoder.get_segment_ids(target_tokens, padding_mask=target_tokens.eq(self.pad))
        tgt_segment_mask = get_segment_mask(tgt_segment_ids)
        tgt_sent_lens = get_sent_lens(tgt_segment_mask)

        target_tokens = expand_segment_dim(tgt_segment_mask, target_tokens)

        assert target_tokens.size(0) == tgt_sent_lens.sum()

        target_tokens = pad_to_new_tensor(target_tokens, tgt_sent_lens, pad=self.pad)

        # normal glancing function
        bsz, pre_len = prev_output_tokens.size()

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)
        nonspecial_mask = prev_output_mask & prev_output_tokens.ne(self.bos) & prev_output_tokens.ne(self.eos)

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

        scatter_to_tensor(prev_output_tokens_clone, glat_prev_output_tokens, prev_segment_ids, prev_sent_lens)

        return prev_output_tokens_clone, glat_info, best_alignment, scattered_predictions

    @torch.no_grad()
    def get_best_alignment(self, encoder_out, prev_output_tokens, target_tokens):
        emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        if FULL_PRECISION:
            emission_scores = emission_scores.float()

        transit_lprobs = self.compute_transit_lprobs(features, prev_output_tokens)

        # convert the prev_output_tokens and emission_scores into segment version
        prev_segment_ids = self.decoder.get_segment_ids(
            prev_output_tokens, padding_mask=prev_output_tokens.eq(self.pad)
        )
        prev_segment_mask = get_segment_mask(prev_segment_ids)
        prev_sent_lens = get_sent_lens(prev_segment_mask)

        emission_scores, prev_output_tokens = expand_segment_dim(
            prev_segment_mask, emission_scores, prev_output_tokens
        )

        assert prev_output_tokens.size(0) == emission_scores.size(0) == prev_sent_lens.sum()

        emission_scores = pad_to_new_tensor(emission_scores, prev_sent_lens, pad=0)
        prev_output_tokens = pad_to_new_tensor(prev_output_tokens, prev_sent_lens, pad=self.pad)

        # convert the target_tokens into segment version
        tgt_segment_ids = self.decoder.get_segment_ids(target_tokens, padding_mask=target_tokens.eq(self.pad))
        tgt_segment_mask = get_segment_mask(tgt_segment_ids)
        tgt_sent_lens = get_sent_lens(tgt_segment_mask)

        target_tokens = expand_segment_dim(tgt_segment_mask, target_tokens)

        assert target_tokens.size(0) == tgt_sent_lens.sum()

        target_tokens = pad_to_new_tensor(target_tokens, tgt_sent_lens, pad=self.pad)

        # normal glancing function
        bsz, pre_len = prev_output_tokens.size()

        target_mask = target_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)
        nonspecial_mask = prev_output_mask & prev_output_tokens.ne(self.bos) & prev_output_tokens.ne(self.eos)

        emission_lprobs = F.log_softmax(emission_scores, dim=-1)
        emission_lprobs = emission_lprobs.gather(2, target_tokens[:, None, :].expand(-1, pre_len, -1))

        best_alignment = get_best_alignment(emission_lprobs, transit_lprobs, prev_output_mask, target_mask)

        return best_alignment

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        _ret = {"output_logs": {}}
        # length prediction
        if self.cfg.length_loss_factor > 0:
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

        # convert the prev_output_tokens and emission_scores into segment version
        prev_segment_ids = self.decoder.get_segment_ids(
            prev_output_tokens, padding_mask=prev_output_tokens.eq(self.pad)
        )
        prev_segment_mask = get_segment_mask(prev_segment_ids)
        prev_sent_lens = get_sent_lens(prev_segment_mask)

        emission_scores, prev_output_tokens = expand_segment_dim(
            prev_segment_mask, emission_scores, prev_output_tokens
        )

        assert emission_scores.size(0) == prev_output_tokens.size(0) == prev_sent_lens.sum()

        emission_scores = pad_to_new_tensor(emission_scores, prev_sent_lens, pad=0)
        prev_output_tokens = pad_to_new_tensor(prev_output_tokens, prev_sent_lens, pad=self.pad)

        # convert the target_tokens into segment version
        tgt_segment_ids = self.decoder.get_segment_ids(tgt_tokens, padding_mask=tgt_tokens.eq(self.pad))
        tgt_segment_mask = get_segment_mask(tgt_segment_ids)
        tgt_sent_lens = get_sent_lens(tgt_segment_mask)

        tgt_tokens = expand_segment_dim(tgt_segment_mask, tgt_tokens)

        assert tgt_tokens.size(0) == tgt_sent_lens.sum()

        tgt_tokens = pad_to_new_tensor(tgt_tokens, tgt_sent_lens, pad=self.pad)

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
            numerator = self._compute_dacrf_numerator(_emission_scores, None, tgt_tokens, target_masks)
            denominator = self._compute_dacrf_normalizer(_emission_scores, None, tgt_tokens, target_masks)

            # if the below condition is not met, the best_alignment would contain errors.
            valid_masks = prev_output_masks.sum(-1) >= target_masks.sum(-1)
            dacrf_loss = -(numerator[valid_masks] - denominator[valid_masks])
            dacrf_loss = dacrf_loss / target_masks[valid_masks].type_as(dacrf_loss).sum(-1)
            dacrf_loss = dacrf_loss.masked_fill(dacrf_loss <= 0, 0)

            dacrf_loss = dacrf_loss.mean() * self.cfg.dacrf_loss_factor

            ret["dacrf_loss"] = {"loss": dacrf_loss, "factor": self.cfg.dacrf_loss_factor}

        ret.update(**_ret)

        return ret

    def _inference_viterbi(self, encoder_out, prev_output_tokens, tgt_tokens):
        prev_output_tokens_clone = torch.full_like(prev_output_tokens, self.pad)

        emission_scores, features = self.extract_features(encoder_out, prev_output_tokens)
        if FULL_PRECISION:
            emission_scores = emission_scores.float()

        transit_lprobs = self.compute_transit_lprobs(features, prev_output_tokens)

        # convert the prev_output_tokens and emission_scores into segment version
        prev_segment_ids = self.decoder.get_segment_ids(
            prev_output_tokens, padding_mask=prev_output_tokens.eq(self.pad)
        )
        prev_segment_mask = get_segment_mask(prev_segment_ids)
        prev_sent_lens = get_sent_lens(prev_segment_mask)

        emission_scores, prev_output_tokens = expand_segment_dim(
            prev_segment_mask, emission_scores, prev_output_tokens
        )

        assert emission_scores.size(0) == prev_output_tokens.size(0) == prev_sent_lens.sum()

        emission_scores = pad_to_new_tensor(emission_scores, prev_sent_lens, pad=0)
        prev_output_tokens = pad_to_new_tensor(prev_output_tokens, prev_sent_lens, pad=self.pad)

        assert prev_output_tokens.size(0) == tgt_tokens.size(0)

        # target_tokens are already converted
        tgt_len = tgt_tokens.size(1)
        pre_len = prev_output_tokens.size(1)

        # normal viterbi decoding
        target_mask = tgt_tokens.ne(self.pad)
        prev_output_mask = prev_output_tokens.ne(self.pad)

        invalid_batch_mask = (prev_output_mask.sum(-1) < target_mask.sum(-1)).bool()  # noqa
        if invalid_batch_mask.any():
            if tgt_len < pre_len:
                target_mask[invalid_batch_mask] = prev_output_mask[invalid_batch_mask][:, :tgt_len]
            else:
                target_mask[invalid_batch_mask] = False

                valid_target_mask = prev_output_mask[invalid_batch_mask]
                valid_target_mask = torch.cat(
                    [
                        valid_target_mask,
                        valid_target_mask.new_full(
                            (valid_target_mask.size(0), tgt_len - pre_len), fill_value=False, dtype=torch.bool
                        ),
                    ],
                    dim=1,
                )

                target_mask[invalid_batch_mask] = valid_target_mask

        emission_lprobs = F.log_softmax(emission_scores, dim=-1)

        output_scores, output_tokens = emission_lprobs.max(dim=-1)

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
            output_tokens, output_scores = self._crf_viterbi_decode(_emission_scores, None, masks=target_mask)
        else:
            output_tokens = output_tokens.gather(1, best_alignment)
            output_scores = output_scores.gather(1, best_alignment)

        scatter_to_tensor(prev_output_tokens_clone, output_tokens, prev_segment_ids, target_mask.sum(-1))

        output_scores = output_scores.new_full(prev_output_tokens_clone.size(), 1.0)

        return prev_output_tokens_clone, output_scores


class GroupDACRFTransformerEncoder(NATransformerEncoder):
    def build_encoder_layer(self, cfg, layer=None, layer_idx=None):
        start_ctxlayer = cfg.encoder.layers - cfg.encoder_ctxlayers
        return GTransformerEncoderLayer(cfg, add_global_attn=(layer_idx >= start_ctxlayer) or cfg.no_encoder_local)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        new_out = super().reorder_encoder_out(encoder_out, new_order)

        segment_ids = encoder_out["segment_ids"]
        if len(segment_ids) > 0:
            for idx, ids in enumerate(segment_ids):
                segment_ids[idx] = ids.index_select(1, new_order)

        new_out["segment_ids"] = segment_ids
        return new_out

    def get_segment_ids(self, tokens, padding_mask):
        segment_ids = tokens.eq(self.dictionary.bos()).cumsum(dim=-1)
        segment_ids = segment_ids.masked_fill(padding_mask, -1)
        return segment_ids

    def forward(self, src_tokens, src_lengths=None, return_fc=False, return_all_hiddens=False, token_embeddings=None):
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        if self.length_token is not None:  # prepend a special length token embedding
            length_embed = self.length_token(src_tokens.new_zeros(src_tokens.size(0), 1))
            x = torch.cat([length_embed, x], dim=1)
            encoder_padding_mask = torch.cat(
                [encoder_padding_mask.new_zeros(src_tokens.size(0), 1), encoder_padding_mask], dim=1
            )

        # account for padding while computing the representation
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        attns = []
        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x if self.length_token is None else x[1:])

        segment_ids = self.get_segment_ids(src_tokens, padding_mask=encoder_padding_mask)

        attn_mask = segment_ids.unsqueeze(2) != segment_ids.unsqueeze(1)
        attn_mask = attn_mask.to(dtype=x.dtype)
        # pad token can attend all, otherwise cause nan problems.
        attn_mask[encoder_padding_mask] = 0

        # encoder layers
        for layer in self.layers:
            x, layer_attn, fc_result = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                attn_mask=attn_mask,
                return_fc=return_fc,
                need_attn=self.cfg.encoder.return_attns,
                need_head_weights=self.cfg.encoder.return_head_attns,
            )

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                if return_fc:
                    fc_results.append(fc_result)
            if self.cfg.encoder.return_attns:
                attns.append(layer_attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x] if self.length_token is None else [x[1:], x[0]],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]  # B x T
            if self.length_token is None
            else [encoder_padding_mask[:, 1:]],
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
            "encoder_attns": attns,  # List[T x B x C]
            "segment_ids": [segment_ids],
        }


class GroupDACRFTransformerDecoder(NATransformerDecoder):
    def build_decoder_layer(self, cfg, no_encoder_attn=False, layer=None, layer_idx=None):
        start_dec_ctxlayer = cfg.decoder.layers - cfg.decoder_ctxlayers
        start_crs_ctxlayer = cfg.decoder.layers - cfg.cross_ctxlayers
        return GTransformerDecoderLayer(
            cfg,
            no_encoder_attn,
            add_self_global_attn=(layer_idx >= start_dec_ctxlayer) or cfg.no_decoder_local,
            add_cross_global_attn=(layer_idx >= start_crs_ctxlayer) or cfg.no_cross_local,
        )

    def get_segment_ids(self, tokens, padding_mask):
        segment_ids = tokens.eq(self.dictionary.bos()).cumsum(dim=-1)
        segment_ids = segment_ids.masked_fill(padding_mask, -1)
        return segment_ids

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        if self.cfg.length_pred_method == "mean":
            enc_feats = encoder_out["encoder_out"][0].transpose(0, 1)  # B x T x C

            segment_mask = get_segment_mask(encoder_out["segment_ids"][0])
            src_lengs = get_sent_lens(segment_mask)
            enc_feats = expand_segment_dim(segment_mask, enc_feats)

            assert enc_feats.size(0) == src_lengs.sum()

            enc_feats = torch.split(enc_feats, src_lengs.tolist(), dim=0)
            enc_feats = torch.stack([feat.mean(dim=0) for feat in enc_feats])

        if self.cfg.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None, beam_size=1):
        src_segment_mask = get_segment_mask(encoder_out["segment_ids"][0])
        src_lengs = get_sent_lens(src_segment_mask)

        if tgt_tokens is not None:
            # obtain the length target
            tgt_segment_ids = self.get_segment_ids(tgt_tokens, padding_mask=tgt_tokens.eq(self.pad))
            tgt_segment_mask = get_segment_mask(tgt_segment_ids)
            tgt_lengs = get_sent_lens(tgt_segment_mask)

            if self.cfg.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + self.embed_length.num_embeddings // 2
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=self.embed_length.num_embeddings - 1)

        else:
            pred_lengs = length_out.topk(beam_size, -1)[1]
            if self.cfg.pred_length_offset:
                length_tgt = pred_lengs - self.embed_length.num_embeddings // 2 + src_lengs[:, None]
            else:
                length_tgt = pred_lengs

        return length_tgt

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        return_all_hiddens=False,
        layers=None,
        source_segment_ids: Optional[torch.Tensor] = None,
        target_segment_ids: Optional[torch.Tensor] = None,
        **unused
    ):
        if embedding_copy:
            x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens, self.forward_copying_source(encoder_out, prev_output_tokens)
            )

        else:
            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attns = []
        cross_attns = []
        inner_states = []
        if return_all_hiddens:
            inner_states.append(x)

        # decoder layers
        for i, layer in enumerate(self.layers if layers is None else layers):
            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, *_ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0)
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                need_attn=self.cfg.decoder.return_attns,
                need_head_weights=self.cfg.decoder.return_head_attns,
                source_segment_ids=source_segment_ids,
                target_segment_ids=target_segment_ids,
            )
            if return_all_hiddens:
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attns": attns, "cross_attns": cross_attns, "inner_states": inner_states}


class GTransformerEncoderLayer(TransformerEncoderLayerBase):
    def __init__(self, cfg, add_global_attn=False):
        super().__init__(cfg)
        self.add_global_attn = add_global_attn
        if self.add_global_attn:
            self.self_attn_global = self.build_self_attention(self.embed_dim, cfg)
            self.self_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())

        if cfg.no_encoder_local:
            self.self_attn = None
            self.self_attn_gate = None

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        return_fc: bool = False,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        attns = {}

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), float("-inf"))

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if not self.cfg.no_encoder_local:
            x_local, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=need_attn and self.need_attn,
                need_head_weights=need_head_weights,
                attn_mask=attn_mask,
            )

        if self.add_global_attn:
            x_global, attn = self.self_attn_global(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=need_attn and self.need_attn,
                need_head_weights=need_head_weights,
                attn_mask=None,
            )

        if self.cfg.no_encoder_local:
            x = x_global
        else:
            if self.add_global_attn:
                # merge with local
                g = self.self_attn_gate(torch.cat([x_local, x_global], dim=-1))
                x = x_local * g + x_global * (1 - g)
            else:
                x = x_local

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        fc_result = x

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if return_fc and not torch.jit.is_scripting():
            return x, attns, fc_result
        return x, attns, None


class GTransformerDecoderLayer(TransformerDecoderLayerBase):
    def __init__(
        self,
        cfg,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
        add_self_global_attn=False,
        add_cross_global_attn=False,
    ):
        super().__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.add_self_global_attn = add_self_global_attn
        self.add_cross_global_attn = add_cross_global_attn

        if self.add_self_global_attn:
            self.self_attn_global = self.build_self_attention(
                self.embed_dim, cfg, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn
            )
            self.self_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())

        if self.add_cross_global_attn:
            self.encoder_attn_global = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())

        if cfg.no_decoder_local:
            self.self_attn = None
            self.self_attn_gate = None

        if cfg.no_cross_local:
            self.encoder_attn = None
            self.encoder_attn_gate = None

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        source_segment_ids: Optional[torch.Tensor] = None,
        target_segment_ids: Optional[torch.Tensor] = None,
    ):
        attns = {}

        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if not self.cfg.no_decoder_local:
            # local attention should use constrained mask
            local_attn_mask = (target_segment_ids.unsqueeze(2) != target_segment_ids.unsqueeze(1)).to(dtype=x.dtype)
            local_attn_mask = local_attn_mask.masked_fill(local_attn_mask.to(torch.bool), float("-inf"))
            local_attn_mask[self_attn_padding_mask] = 0

            x_local, self_attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=need_attn and self.need_attn,
                need_head_weights=need_head_weights,
                attn_mask=local_attn_mask,
            )

        if self.add_self_global_attn:
            x_global, self_attn = self.self_attn_global(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=need_attn and self.need_attn,
                need_head_weights=need_head_weights,
                attn_mask=None,
            )

        if self.cfg.no_decoder_local:
            x = x_global
        else:
            if self.add_self_global_attn:
                # merge with local
                g = self.self_attn_gate(torch.cat([x_local, x_global], dim=-1))
                x = x_local * g + x_global * (1 - g)
            else:
                x = x_local

        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        if not self.cfg.no_cross_local:
            cross_attn_mask = (target_segment_ids.unsqueeze(2) != source_segment_ids.unsqueeze(1)).to(dtype=x.dtype)
            cross_attn_mask = cross_attn_mask.masked_fill(cross_attn_mask.to(torch.bool), float("-inf"))
            cross_attn_mask[self_attn_padding_mask] = 0

            x_local, cross_attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn and self.need_attn,
                need_head_weights=need_head_weights,
                attn_mask=cross_attn_mask,
            )

        if self.add_cross_global_attn:
            x_global, cross_attn = self.encoder_attn_global(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn and self.need_attn,
                need_head_weights=need_head_weights,
                attn_mask=None,
            )

        if self.cfg.no_cross_local:
            x = x_global
        else:
            if self.add_cross_global_attn:
                # merge with local
                g = self.encoder_attn_gate(torch.cat([x_local, x_global], dim=-1))
                x = x_local * g + x_global * (1 - g)
            else:
                x = x_local

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attns, self_attn_state
        return x, attns, None


@register_model_architecture("dacrf_transformer_doc", "dacrf_transformer_doc_wmt_en_de")
def dacrf_transformer_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("dacrf_transformer_doc", "dacrf_transformer_doc_iwslt_de_en")
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


@register_model_architecture("dacrf_transformer_doc", "dacrf_transformer_doc_big")
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
