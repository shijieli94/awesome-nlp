##########################################################################
# Copyright (C) 2022 COAI @ Tsinghua University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

import logging
import time
from collections import namedtuple
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.dataclass import ChoiceEnum
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.nonautoregressive_transformer import (
    NATransformerConfig,
    NATransformerDecoder,
    NATransformerModel,
)
from fairseq.modules.positional_embedding import PositionalEmbedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from torch import Tensor, nn

DecodeResult = namedtuple("DecodeResult", ["future", "fn", "args"])

LinksPositionChoices = ChoiceEnum(["none", "sine", "learned"])
DecodeStrategyChoices = ChoiceEnum(["greedy", "lookahead", "viterbi", "jointviterbi", "sample", "beamsearch"])
GlancingSamplingChoices = ChoiceEnum(["random", "number-random", "cmlm", "fix"])

logger = logging.getLogger(__name__)


# @jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    # Solving nan issue when x contains -inf
    # See https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim, keepdim=True)
    mask = m == float("-inf")
    m = m.detach()
    s = (x - m.masked_fill(mask, 0)).exp().sum(dim=dim, keepdim=True)
    return s.masked_fill(mask, 1).log() + m.masked_fill(mask, float("-inf"))


# @jit.script
def loop_function_noempty(last_f: Tensor, links: Tensor, match: Tensor) -> Tensor:
    f_next = logsumexp(last_f + links, dim=1)  # batch * 1 * pre_len
    return match + f_next.transpose(1, 2)  # batch * pre_len * 1


# @jit.script
def loop_function_noempty_max(last_f: Tensor, links: Tensor, match: Tensor) -> Tensor:
    f_next = torch.max(last_f + links, dim=1)[0]  # batch * 1 * pre_len
    return match + f_next.unsqueeze(-1)  # batch * pre_len * 1


def load_logsoftmax_gather_inplace():
    try:
        from dag_search import logsoftmax_gather_inplace

        return logsoftmax_gather_inplace, True

    except ImportError:

        def logsoftmax_gather_inplace(word_ins_out, select_idx):
            r"""Fused operation of log_softmax and gather"""
            logits = torch.log_softmax(word_ins_out, -1, dtype=torch.float32)
            match = logits.gather(dim=-1, index=select_idx)
            return word_ins_out, match

        return logsoftmax_gather_inplace, False


def load_dag_loss():
    try:
        from dag_search import dag_loss

        return dag_loss, True

    except ImportError:

        def dag_loss(match_all, links, output_length, target_length):
            r"""
            Pytorch implementation for calculating the dag loss.
            Input:
                match_all (torch.FloatTensor or torch.HalfTensor):
                    Shape: [batch_size, max_target_length, max_output_length]
                    match_all[b, i, j] represents -log P(y_i| v_j), the probability of predicting the i-th token in the reference
                    based on the j-th vertex.
                    (Note: float32 are preferred; float16 may cause precision problem)
                links (torch.FloatTensor or torch.HalfTensor):
                    Shape: [batch_size, max_output_length, max_transition_length]
                    links[b, i, j] represents the transition probability from the i-th vertex to **the j-th vertex**.
                    (Note: this parameter is different from the cuda version)
                output_length (torch.LongTensor):
                    Shape: [batch_size]
                    output_length should be the graph size, the vertices (index >= graph size) are ignored
                target_length (torch.LongTensor):
                    Shape: [batch_size]
                    target_length is the reference length, the tokens (index >= target length) are ignored

            Output (torch.FloatTensor or torch.HalfTensor):
                Shape: [batch_size]
                the loss of each sample
            """
            match_all = match_all.transpose(1, 2)
            bsz, pre_len, tgt_len = match_all.size()
            assert links.shape[1] == links.shape[2] == pre_len, "links should be batch_size * pre_len * pre_len"

            f_arr = []
            f_init = match_all.new_full((bsz, pre_len, 1), float("-inf"))
            f_init[:, 0, 0] = match_all[:, 0, 0]
            f_arr.append(f_init)

            match_all_chunk = torch.chunk(match_all, tgt_len, -1)  # k * [batch * pre_len * 1]

            for k in range(1, tgt_len):
                f_now = loop_function_noempty(f_arr[-1], links, match_all_chunk[k])
                f_arr.append(f_now)

            loss_result = torch.cat(f_arr, -1)[range(bsz), output_length - 1, target_length - 1]

            return loss_result

        return dag_loss, False


def load_best_alignment():
    try:
        from dag_search import best_alignment

        return best_alignment, True

    except ImportError:

        def __max_loss(match_all, links, output_length, target_length):
            match_all = match_all.transpose(1, 2)
            bsz, pre_len, tgt_len = match_all.size()
            assert links.shape[1] == links.shape[2] == pre_len, "links should be batch_size * pre_len * pre_len"

            f_arr = []
            f_init = match_all.new_full((bsz, pre_len, 1), float("-inf"))
            f_init[:, 0, 0] = match_all[:, 0, 0]
            f_arr.append(f_init)

            match_arr = torch.chunk(match_all, tgt_len, -1)
            for i in range(1, tgt_len):
                f_now = loop_function_noempty_max(f_arr[-1], links, match_arr[i])
                f_arr.append(f_now)

            # only select the last cumulative lprobs
            all_lprobs = torch.cat(f_arr, -1)[range(bsz), output_length - 1, target_length - 1]

            return all_lprobs

        def dag_best_alignment(match_all, links, output_length, target_length):
            r"""
            Function to obtain the alignment between prediction and reference
            Input:
                match_all (torch.FloatTensor or torch.HalfTensor):
                    Shape: [batch_size, max_target_length, max_output_length]
                    match_all[b, i, j] represents -log P(y_i| v_j), the probability of predicting the i-th token in the reference
                    based on the j-th vertex.
                    (Note: float32 are preferred; float16 may cause precision problem)
                links (torch.FloatTensor or torch.HalfTensor):
                    Shape: [batch_size, max_output_length, max_transition_length]
                    links[b, i, j] represents the transition probability from the i-th vertex to **the j-th vertex**.
                    (Note: this parameter is different from the cuda version)
                output_length (torch.LongTensor):
                    Shape: [batch_size]
                    output_length should be the graph size, the vertices (index >= graph size) are ignored
                target_length (torch.LongTensor):
                    Shape: [batch_size]
                    target_length is the reference length, the tokens (index >= target length) are ignored

            Output (torch.LongTensor):
                Shape: [batch_size, max_output_length]
                if output[b, i]>=0, it represents the index of target token aligned with the i-th vertex
                otherwise, output[b, i] = -1, it represents the i-th vertex is not aligned with any target token
            """
            with torch.enable_grad():
                match_all.requires_grad_()
                all_lprobs = __max_loss(match_all, links, output_length, target_length)
                match_grad = torch.autograd.grad(all_lprobs.sum(), [match_all])[0]  # batch * tgt_len * pre_len
            path_value, path = match_grad.max(dim=1)
            path.masked_fill_(path_value < 0.5, -1)
            return path

        return dag_best_alignment, False


# Due to the use of multi-processing, beamsearch functions are in global scope
def init_beam_search(*args):
    from dag_search import beam_search_init

    beam_search_init(*args)


def call_dag_search(*args):
    from dag_search import dag_search

    res, score = dag_search(*args)
    output_tokens = torch.tensor(res)
    output_scores = torch.tensor(score).unsqueeze(-1).expand_as(output_tokens)
    return output_tokens, output_scores


def subprocess_init(n):
    time.sleep(10)  # Do something to wait all subprocess to start
    print(f"overlapped decoding: subprocess {n} initializing", flush=True)
    return n


def inference_post_process(inference_result, decoder_out):  # post process after inference for overlapped decoding
    device = decoder_out.output_tokens.device
    output_tokens, output_scores = inference_result
    return decoder_out._replace(
        output_tokens=output_tokens.to(device),
        output_scores=output_scores.to(device),
        attn=None,
        history=None,
    )


@dataclass
class DATransformerModelConfig(NATransformerConfig):
    # common configs
    links_position: LinksPositionChoices = field(
        default="learned", metadata={"help": "If adding positional embeddings to predict transitions"}
    )
    max_transition_length: int = field(
        default=99999,
        metadata={
            "help": "Specifies the maximum transition distance. A value of -1 indicates no limit, "
            "but this cannot be used with CUDA custom operations. "
            "To use CUDA operations with no limit, specify a very large number such as 99999."
        },
    )
    decode_strategy: DecodeStrategyChoices = field(default="lookahead", metadata={"help": "Decoding strategy to use."})
    upsample_scale: str = field(
        default="4-8",
        metadata={
            "help": "Specifies the upsample scale for the decoder input length in training. "
            'For instance, "4~8" indicates that the upsampling scale will be uniformly sampled from the range [4, 8];'
            '"4" indicates fixed upsampling scale.'
        },
    )
    decode_upsample_scale: int = field(
        default=8,
        metadata={
            "help": "Up-sampling scale to determine the decoder size during inference. "
            "If --upsample-scale used in training is a fixed number, this parameter should be the same value."
            "If --upsample-scale used in training is a range, this parameter can be the average of the range, or tuned on the validation set."
        },
    )

    # glancing transformer
    glance_p: Optional[str] = field(
        default=None,
        metadata={
            "help": "Set the glancing probability and its annealing schedule. "
            "For example, '0.5@0-0.1@200k' indicates annealing probability from 0.5 to 0.1 in the 0-200k steps."
        },
    )
    glance_strategy: GlancingSamplingChoices = field(
        default="number-random", metadata={"help": "Set the glancing strategy."}
    )
    no_force_emit: bool = field(
        default=False,
        metadata={"help": "If true, the position of glanced tokens in the second forward pass will not be fixed."},
    )

    # inference
    no_repeat_ngram_size: int = field(
        default=0,
        metadata={
            "help": "Prevent repeated k-grams (not necessarily consecutive) with order n or higher in the generated text. Use 0 to disable this feature. "
            "This argument is used in lookahead, sample, and beam search decoding methods."
        },
    )
    no_consecutive_repeat_ngram_size: int = field(
        default=0,
        metadata={
            "help": "Prevent consecutive repeated k-grams (k <= n) in the generated text. Use 0 to disable this feature. "
            "This argument is used in greedy, lookahead, sample, and beam search decoding methods."
        },
    )
    decode_beta: float = field(
        default=1,
        metadata={
            "help": "Parameter used to scale the score of logits in beamsearch decoding. "
            "The score of a sentence is given by: sum P(y_i|a_i) + beta * sum log(a_i|a_{i-1})"
        },
    )
    decode_topn: int = field(
        default=5,
        metadata={
            "help": "Number of top candidates to consider during transition. "
            "This argument is used in lookahead decoding with n-gram prevention, and sample and beamsearch decoding methods."
        },
    )
    decode_topp: float = field(
        default=0.9,
        metadata={
            "help": "Maximum probability of top candidates to consider during transition. "
            "This argument is used in lookahead decoding with n-gram prevention, and sample and beamsearch decoding methods."
        },
    )

    viterbi_lenpen: float = field(
        default=1,
        metadata={
            "help": "Parameter used for length penalty in Viterbi decoding. The sentence with the highest score is found using: P(A, Y|X) / |Y|^{beta}"
        },
    )


@register_model("dag_transformer", dataclass=DATransformerModelConfig)
class DATransformerModel(NATransformerModel):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.upsample_scale = list(map(int, cfg.upsample_scale.split("-")))
        self.glat_scheduler = utils.get_anneal_argument_parser(cfg.glance_p) if cfg.glance_p is not None else None
        if cfg.decode_strategy == "beamsearch":
            self.init_beam_search()

    def init_beam_search(self):
        initargs = (
            self.cfg.decode_max_batchsize,
            self.cfg.decode_beamsize,
            self.cfg.decode_top_cand_n,
            self.decoder.max_positions(),
            self.cfg.max_decoder_batch_tokens,
            self.cfg.decode_threads_per_worker,
            self.tgt_dict,
            self.cfg.decode_lm_path,
        )

        if self.cfg.decode_max_workers >= 1:  # overlapped decoding
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            self.executor = ProcessPoolExecutor(
                max_workers=self.cfg.decode_max_workers,
                mp_context=ctx,
                initializer=init_beam_search,
                initargs=initargs,
            )
            for x in self.executor.map(subprocess_init, range(self.cfg.decode_max_workers)):
                pass
        else:  # vanilla decoding
            init_beam_search(*initargs)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = DATransformerDecoder(cfg, tgt_dict, embed_tokens)
        if cfg.apply_bert_init:
            decoder.apply(init_bert_params)
        return decoder

    def _compute_dag_loss(self, outputs, prev_output_tokens, targets, links):
        pre_len = prev_output_tokens.shape[1]

        target_length = targets.ne(self.pad).sum(dim=-1)
        output_length = prev_output_tokens.ne(self.pad).sum(dim=-1)

        match = load_logsoftmax_gather_inplace()[0](outputs, targets.unsqueeze(1).expand(-1, pre_len, -1))[
            1
        ].transpose(1, 2)

        # verify the loss sum on all fragments is correct
        # assert model.cfg.max_transition_length != -1
        # tmp = self._brute_force_fragment_sum(match, force_emit, model.restore_valid_links(links))

        # if match_mask is not None and not self.cfg.no_force_emit:
        #     glat_prev_mask = keep_word_mask.unsqueeze(1)
        #     match = match.masked_fill(glat_prev_mask, 0) + match.masked_fill(~match_mask, float("-inf")).masked_fill(~glat_prev_mask, 0).detach()

        dag_loss, use_cuda = load_dag_loss()
        if use_cuda:
            assert (
                self.cfg.max_transition_length != -1
            ), "cuda dag loss does not support max_transition_length=-1. You can use a very large number such as 99999"
            loss_result = dag_loss(match, links, output_length, target_length)
        else:
            if self.cfg.max_transition_length != -1:
                links = self.restore_valid_links(links)
            loss_result = -dag_loss(match, links, output_length, target_length)

        invalid_masks = loss_result.isinf() | loss_result.isnan()
        if invalid_masks.sum() > 0:
            logger.warning(f"{invalid_masks.sum()} samples have nan or inf in computed loss.")
            loss_result = loss_result.masked_fill(invalid_masks, 0)

        return (loss_result / target_length).mean()

    @torch.no_grad()
    def glancing_sampling(self, encoder_out, prev_output_tokens, tgt_tokens, ratio):
        with utils.model_eval(self.decoder):
            word_ins_out, links = self.decoder(encoder_out, prev_output_tokens, require_links=True)

        bsz, pre_len, _ = links.size()

        target_lengths = tgt_tokens.ne(self.pad).sum(1)
        output_lengths = prev_output_tokens.ne(self.pad).sum(1)

        pred_tokens = word_ins_out.argmax(-1)
        match = load_logsoftmax_gather_inplace()[0](word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, pre_len, -1))[
            1
        ].transpose(1, 2)

        best_alignment, use_cuda = load_best_alignment()

        if use_cuda:
            assert self.cfg.max_transition_length != -1, (
                "cuda dag best alignment does not support max_transition_length=-1. "
                "You can use a very large number such as 99999"
            )
            path = best_alignment(match, links, output_lengths, target_lengths)  # batch * pre_len

        else:
            if self.cfg.max_transition_length != -1:
                links = self.restore_valid_links(links)
            path = best_alignment(match, links, output_lengths, target_lengths)

        predict_assigned_mask = path >= 0
        oracle = tgt_tokens.gather(-1, path.clip(min=0))  # bsz * pre_len
        same_num = ((pred_tokens == oracle) & predict_assigned_mask).sum(1)

        if self.cfg.glance_strategy == "random":
            keep_probs = ((target_lengths - same_num) / target_lengths * ratio).unsqueeze(-1) * predict_assigned_mask

        else:  # exactly sample glance_nums tokens
            probs = torch.rand_like(oracle, dtype=torch.float).masked_fill_(~predict_assigned_mask, -100)

            if self.cfg.glance_strategy == "number-random":
                glance_nums = ((target_lengths - same_num) * ratio + 0.5).long()

            elif self.cfg.glance_strategy == "cmlm":
                glance_nums = (target_lengths * torch.rand_like(target_lengths, dtype=torch.float) + 0.5).long()

            elif self.cfg.glance_strategy == "fix":
                glance_nums = (target_lengths * ratio + 0.5).long()

            probs_thresh = probs.sort(descending=True)[0].gather(-1, glance_nums.unsqueeze(-1))
            keep_probs = (probs > probs_thresh).type_as(probs)

        keep_mask = torch.rand_like(prev_output_tokens, dtype=torch.float) < keep_probs

        glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_mask, 0) + oracle.masked_fill(~keep_mask, 0)

        force_emit = path.masked_fill(~keep_mask, -1)

        glat_info = {
            "_glat@total": utils.item(target_lengths.sum()),
            "_glat@n_correct": utils.item(same_num.sum()),
            "force_emit": force_emit,
        }

        return glat_prev_output_tokens, glat_info

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        ret = {"length": {"out": length_out, "tgt": length_tgt, "factor": self.cfg.length_loss_factor}}

        # graph size is larger than target length
        graph_size = src_lengths * torch.randint(
            low=self.upsample_scale[0], high=self.upsample_scale[1] + 1, size=(1,)
        ).to(src_lengths)
        prev_output_tokens = self._initialize_output_tokens_with_length(
            encoder_out, src_tokens, graph_size
        ).output_tokens

        # decoding
        glat_ratio = None if self.glat_scheduler is None else max(0, self.glat_scheduler(self.get_num_updates()))
        if glat_ratio:
            prev_output_tokens, glat_infos = self.glancing_sampling(
                encoder_out, prev_output_tokens, tgt_tokens, ratio=glat_ratio
            )
            ret["output_logs"] = glat_infos

        word_ins_out, links = self.decoder(encoder_out, prev_output_tokens, require_links=True)

        if self.cfg.compute_ngram_loss:
            dag_loss = self._compute_ngram_loss(word_ins_out, prev_output_tokens, tgt_tokens, links)
        else:
            dag_loss = self._compute_dag_loss(word_ins_out, prev_output_tokens, tgt_tokens, links)

        ret["dag_loss"] = {"loss": dag_loss}

        return ret

    def forward_decoder(self, decoder_out, encoder_out, src_tokens, decoding_graph=False, **kwargs):
        src_lengths = src_tokens.ne(self.pad).sum(1)
        output_tokens = self._initialize_output_tokens_with_length(
            encoder_out, src_tokens, src_lengths * self.cfg.decode_upsample_scale
        ).output_tokens

        decoder_out = decoder_out._replace(output_tokens=output_tokens)

        output_logits, links = self.decoder(encoder_out, output_tokens, require_links=True)

        if self.cfg.max_transition_length != -1:
            links = self.restore_valid_links(links)

        result = self.inference(decoder_out, output_logits, links)

        if not decoding_graph:
            return result

        if isinstance(result, DecodeResult):
            hypos_result = result.future.result()
            for fn, args in zip(result.fn, result.args):
                hypos_result = fn(hypos_result, *args)
            result = hypos_result
        return result, self._analyze_graph(result.output_tokens, output_tokens, output_logits, links)

    def inference(self, decoder_out, output_logits, links):
        output_tokens = decoder_out.output_tokens
        output_length = output_tokens.ne(self.pad).sum(dim=-1)

        output_lprobs = output_logits.log_softmax(dim=-1)

        if self.cfg.decode_strategy in ["lookahead", "greedy"]:
            if (
                self.cfg.no_repeat_ngram_size > 0
                or self.cfg.no_consecutive_repeat_ngram_size > 0
                and self.cfg.decode_strategy == "lookahead"
            ):
                inference_result = self.inference_lookahead_repeat_prevent(links, output_lprobs, output_length)
            else:
                inference_result = self.inference_lookahead_simple(links, output_lprobs, output_length)

        elif "viterbi" in self.cfg.decode_strategy:
            assert (
                self.cfg.no_consecutive_repeat_ngram_size == 0 and self.cfg.no_repeat_ngram_size == 0
            ), "viterbi decoding does not support repeated ngram prevention"
            inference_result = self.inference_viterbi(links, output_lprobs, output_length)

        elif self.cfg.decode_strategy == "sample":
            inference_result = self.inference_sample(links, output_lprobs, output_length)

        elif self.cfg.decode_strategy == "beamsearch":
            inference_result = self.inference_beamsearch(links, output_lprobs, output_length)

        if isinstance(inference_result, Future):
            return DecodeResult(future=inference_result, fn=[inference_post_process], args=[(decoder_out,)])
        return inference_post_process(inference_result, decoder_out)

    def inference_lookahead_simple(self, links, output_lprobs, output_length):
        unreduced_lprobs, unreduced_tokens = output_lprobs.max(dim=-1)

        unreduced_tokens = unreduced_tokens.tolist()
        output_length = output_length.tolist()

        if self.cfg.decode_strategy == "lookahead":
            links_idx = (links + unreduced_lprobs.unsqueeze(1) * self.cfg.decode_beta).max(dim=-1)[1].tolist()
        elif self.cfg.decode_strategy == "greedy":
            links_idx = links.max(dim=-1)[1].tolist()

        unpad_output_tokens = []
        for i, length in enumerate(output_length):
            last = unreduced_tokens[i][0]
            j = 0
            res = [last]
            while j != length - 1:
                j = links_idx[i][j]
                now_token = unreduced_tokens[i][j]
                if now_token != self.pad and now_token != last:
                    res.append(now_token)
                last = now_token
            unpad_output_tokens.append(res)

        output_seqlen = max([len(res) for res in unpad_output_tokens])
        output_tokens = [res + [self.pad] * (output_seqlen - len(res)) for res in unpad_output_tokens]
        output_tokens = torch.tensor(output_tokens)
        output_scores = torch.full_like(output_tokens, 1.0, dtype=torch.float)
        return output_tokens, output_scores

    def inference_lookahead_repeat_prevent(self, links, output_lprobs, output_length):
        batch_size, pre_len, _ = links.size()

        top_lprobs, top_lprobs_idx = output_lprobs.topk(self.cfg.decode_topn, dim=-1)

        dag_scores = links.unsqueeze(-1) + top_lprobs.unsqueeze(1) * self.cfg.decode_beta
        dag_scores, top_cand_idx = dag_scores.reshape(batch_size, pre_len, -1).topk(self.cfg.decode_topn, dim=-1)

        next_step_idx = torch.div(top_cand_idx, self.cfg.decode_topn, rounding_mode="floor")

        lprobs_idx_idx = top_cand_idx % self.cfg.decode_topn  # batch * pre_len * top_cand_n

        idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*next_step_idx.shape)
        logits_idx = top_lprobs_idx[idx1, next_step_idx, lprobs_idx_idx]  # batch * pre_len * top_cand_n

        dag_scores = dag_scores.exp().cpu().numpy()
        nextstep_idx = next_step_idx.int().cpu().numpy()
        logits_idx = logits_idx.int().cpu().numpy()
        output_length_cpu = output_length.int().cpu().numpy()

        output_tokens = []
        for i, length in enumerate(output_length_cpu):
            j = 0
            res = [top_lprobs_idx[i][0][0]]
            banned_ngram = set()
            while j != length - 1:
                temp_banned_token = set()
                if res:
                    temp_banned_token.add(res[-1])
                for k in range(2, min(self.args.decode_no_consecutive_repeated_ngram, (len(res) + 1) // 2) + 1, 1):
                    if all([res[-l] == res[-k - l] for l in range(1, k, 1)]):
                        temp_banned_token.add(res[-k])
                prob = 0
                for k, cand in enumerate(logits_idx[i, j]):
                    if (
                        cand not in temp_banned_token
                        and (tuple(res[-self.args.decode_no_repeated_ngram + 1 :]) + (cand,)) not in banned_ngram
                    ):
                        break
                    prob += dag_scores[i, j, k]
                    if prob > self.args.decode_top_p:
                        k, cand = 0, logits_idx[i, j, 0]
                        break
                else:
                    k, cand = 0, logits_idx[i, j, 0]
                j = nextstep_idx[i, j, k]
                res.append(cand)
                if self.args.decode_no_repeated_ngram and len(res) >= self.args.decode_no_repeated_ngram:
                    banned_ngram.add(tuple(res[-self.args.decode_no_repeated_ngram :]))
            output_tokens.append(res)

        output_seqlen = max([len(res) for res in output_tokens])
        output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in output_tokens]
        output_tokens = torch.tensor(output_tokens)
        output_scores = torch.full_like(output_tokens, 1.0, dtype=torch.float)
        return output_tokens, output_scores

    def inference_sample(self, links, output_lprobs, output_length):
        batch_size, prelen, _ = links.shape

        top_logits, top_logits_idx = output_lprobs.topk(self.args.decode_top_cand_n, dim=-1)
        dagscores_arr = (links / self.args.decode_temperature).log_softmax(dim=-1).unsqueeze(
            -1
        ) + top_logits.unsqueeze(
            1
        ) * self.args.decode_beta  # batch * prelen * prelen * top_cand_n
        dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(
            self.args.decode_top_cand_n, dim=-1
        )  # batch * prelen * top_cand_n

        nextstep_idx = torch.div(
            top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor"
        )  # batch * prelen * top_cand_n
        logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n  # batch * prelen * top_cand_n
        idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
        logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx]  # batch * prelen * top_cand_n

        dagscores = dagscores.exp().cpu().numpy()
        nextstep_idx = nextstep_idx.int().cpu().numpy()
        logits_idx = logits_idx.int().cpu().numpy()
        output_length_cpu = output_length.int().cpu().numpy()

        output_tokens = []
        for i, length in enumerate(output_length_cpu):
            j = 0
            res = []
            banned_ngram = set()

            while j != length - 1:
                temp_banned_token = set()
                if res:
                    temp_banned_token.add(res[-1])
                for k in range(2, min(self.args.decode_no_consecutive_repeated_ngram, (len(res) + 1) // 2) + 1, 1):
                    if all([res[-l] == res[-k - l] for l in range(1, k, 1)]):
                        temp_banned_token.add(res[-k])

                problist = []
                realprob = 0
                for k, cand in enumerate(logits_idx[i, j]):
                    realprob += dagscores[i, j, k]
                    if (
                        cand in temp_banned_token
                        or (tuple(res[-self.args.decode_no_repeated_ngram + 1 :]) + (cand,)) in banned_ngram
                    ):
                        problist.append(1e-5)
                    else:
                        problist.append(dagscores[i, j, k])
                    if realprob > self.args.decode_top_p:
                        break
                problist = np.array(problist)
                problist /= problist.sum()
                k = np.random.choice(len(problist), 1, p=problist).item()
                cand = logits_idx[i, j, k]

                j = nextstep_idx[i, j, k]
                res.append(cand)

                if self.args.decode_no_repeated_ngram and len(res) >= self.args.decode_no_repeated_ngram:
                    banned_ngram.add(tuple(res[-self.args.decode_no_repeated_ngram :]))

            output_tokens.append(res)

        output_seqlen = max([len(res) for res in output_tokens])
        output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in output_tokens]
        output_tokens = torch.tensor(output_tokens)
        output_scores = torch.full(output_tokens.size(), 1.0)
        return output_tokens, output_scores

    def inference_beamsearch(self, links, output_logits_normalized, output_length):
        batch_size, prelen, _ = links.shape

        assert (
            batch_size <= self.args.decode_max_batchsize
        ), "Please set --decode-max-batchsize for beamsearch with a larger batch size"

        top_logits, top_logits_idx = output_logits_normalized.topk(self.args.decode_top_cand_n, dim=-1)
        dagscores_arr = (
            links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta
        )  # batch * prelen * prelen * top_cand_n
        dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(
            self.args.decode_top_cand_n, dim=-1
        )  # batch * prelen * top_cand_n

        nextstep_idx = torch.div(
            top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor"
        )  # batch * prelen * top_cand_n
        logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n  # batch * prelen * top_cand_n
        idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
        logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx]  # batch * prelen * top_cand_n

        # rearange_idx = logits_idx.sort(dim=-1)[1]
        # dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
        # nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
        # logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n

        if (
            dagscores.get_device() == -1
            and self.args.decode_strategy == "beamsearch"
            and self.args.decode_max_workers < 1
        ):
            raise RuntimeError(
                "Please specify decode_max_workers at least 1 if you want to run DA-Transformer on cpu while using beamsearch decoding. "
                "It will use a seperate process for beamsearch because the multi-thread library used in PyTorch and DAG-Search is conflict."
            )

        dagscores = np.ascontiguousarray(dagscores.float().cpu().numpy())
        nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
        logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
        output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())

        if self.args.decode_max_workers >= 1:
            future = self.executor.submit(
                call_dag_search,
                dagscores,
                nextstep_idx,
                logits_idx,
                output_length_cpu,
                self.args.decode_alpha,
                self.args.decode_gamma,
                self.args.decode_beamsize,
                self.args.decode_max_beam_per_length,
                self.args.decode_top_p,
                self.tgt_dict.pad_index,
                self.tgt_dict.bos_index,
                1 if self.args.decode_dedup else 0,
                self.args.decode_no_consecutive_repeated_ngram,
                self.args.decode_no_repeated_ngram,
            )
            return future
        else:
            res = call_dag_search(
                dagscores,
                nextstep_idx,
                logits_idx,
                output_length_cpu,
                self.args.decode_alpha,
                self.args.decode_gamma,
                self.args.decode_beamsize,
                self.args.decode_max_beam_per_length,
                self.args.decode_top_p,
                self.tgt_dict.pad_index,
                self.tgt_dict.bos_index,
                1 if self.args.decode_dedup else 0,
                self.args.decode_no_consecutive_repeated_ngram,
                self.args.decode_no_repeated_ngram,
            )
            return res

    def inference_viterbi(self, links, output_logits_normalized, output_length):
        unreduced_logits, unreduced_tokens = output_logits_normalized.max(dim=-1)
        unreduced_tokens = unreduced_tokens.tolist()

        scores = []
        indexs = []
        # batch * graph_length
        alpha_t = links[:, 0]
        if self.args.decode_strategy == "jointviterbi":
            alpha_t += unreduced_logits[:, 0].unsqueeze(1)
        batch_size, graph_length, _ = links.size()
        alpha_t += unreduced_logits
        scores.append(alpha_t)

        # the exact max_length should be graph_length - 2, but we can reduce it to an appropriate extent to speedup decoding
        max_length = int(2 * graph_length / self.args.decode_upsample_scale)
        for i in range(max_length - 1):
            alpha_t, index = torch.max(alpha_t.unsqueeze(-1) + links, dim=1)
            if self.args.decode_strategy == "jointviterbi":
                alpha_t += unreduced_logits
            scores.append(alpha_t)
            indexs.append(index)

        # max_length * batch * graph_length
        indexs = torch.stack(indexs, dim=0)
        scores = torch.stack(scores, dim=0)
        link_last = torch.gather(
            links, -1, (output_length - 1).view(batch_size, 1, 1).repeat(1, graph_length, 1)
        ).view(1, batch_size, graph_length)
        scores += link_last

        # max_length * batch
        scores, max_idx = torch.max(scores, dim=-1)
        lengths = torch.arange(max_length).unsqueeze(-1).repeat(1, batch_size) + 1
        length_penalty = (lengths**self.args.decode_viterbibeta).cuda(scores.get_device())
        scores = scores / length_penalty
        max_score, pred_length = torch.max(scores, dim=0)
        pred_length = pred_length + 1

        initial_idx = torch.gather(max_idx, 0, (pred_length - 1).view(1, batch_size)).view(batch_size).tolist()
        unpad_output_tokens = []
        indexs = indexs.tolist()
        pred_length = pred_length.tolist()
        for i, length in enumerate(pred_length):
            j = initial_idx[i]
            last = unreduced_tokens[i][j]
            res = [last]
            for k in range(length - 1):
                j = indexs[length - k - 2][i][j]
                now_token = unreduced_tokens[i][j]
                if now_token != self.tgt_dict.pad_index and now_token != last:
                    res.insert(0, now_token)
                last = now_token
            unpad_output_tokens.append(res)

        output_seqlen = max([len(res) for res in unpad_output_tokens])
        output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
        output_tokens = torch.tensor(output_tokens)
        output_scores = torch.full(output_tokens.size(), 1.0)
        return output_tokens, output_scores

    @torch.no_grad()
    def _analyze_graph(self, tgt_tokens, output_tokens, logits, links):
        tgt_tokens = tgt_tokens.long()
        target_length = (tgt_tokens != self.tgt_dict.pad_index).sum(dim=-1)
        output_length = (output_tokens != self.tgt_dict.pad_index).sum(dim=-1)
        batch_size, prelen, _ = links.shape

        # calculate node passing probability
        f_arr = []
        f_init = torch.zeros(batch_size, prelen, 1, dtype=links.dtype, device=links.device).fill_(float("-inf"))
        f_init[:, 0, 0].zero_()
        f_arr.append(f_init)
        for _ in range(1, prelen):
            f_now = torch.logsumexp(f_arr[-1] + links, 1, keepdim=True).transpose(1, 2)  # batch * prelen * 1
            f_arr.append(f_now)
        f_arr = torch.cat(f_arr, -1).transpose(1, 2)
        node_pass_prob = f_arr.exp().sum(dim=1).tolist()

        # calculate max path
        word_ins_out, match = load_logsoftmax_gather_inplace()[0](
            logits, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1)
        )
        match = match.transpose(1, 2)
        paths = load_best_alignment()[0](match, links, output_length, target_length).tolist()

        max_paths = []
        for i, raw_path in enumerate(paths):
            sample_max_paths = [-1 for _ in range(target_length[i])]
            for k, v in enumerate(raw_path):
                sample_max_paths[v] = k
            max_paths.append(sample_max_paths)

        # calculate top tokens
        top_k = 5
        val, idx = logits.softmax(dim=-1).topk(k=top_k, dim=-1)
        val = val.tolist()
        idx = idx.tolist()
        node_tokens = []
        node_probs = []
        for i in range(batch_size):
            sample_node_tokens = []
            sample_node_probs = []
            for j in range(output_length[i]):
                sample_node_tokens.append([self.tgt_dict[x] for x in idx[i][j]])
                sample_node_probs.append(val[i][j])
            node_tokens.append(sample_node_tokens)
            node_probs.append(sample_node_probs)

        links = torch.nan_to_num(links.softmax(dim=-1), nan=0).tolist()
        return {
            "node_pass_prob": node_pass_prob,
            "max_paths": max_paths,
            "node_tokens": node_tokens,
            "node_probs": node_probs,
            "links": links,
        }

    def _compute_ngram_loss(self, word_ins_out, prev_output_tokens, tgt_tokens, transition, name="loss", factor=1.0):
        ngrams_order = 4

        probs = word_ins_out.softmax(dim=-1)
        transition = torch.exp(transition)

        bsz, tgt_len = tgt_tokens.size()

        with torch.no_grad():
            tgt_tokens_list = tgt_tokens.tolist()

            ngrams_dict_bsz = [{} for _ in range(bsz)]
            ngrams_list_bsz = [[] for _ in range(bsz)]
            ngrams_max_count_bsz = [[] for _ in range(bsz)]

            for i in range(0, tgt_tokens.size(1) - ngrams_order + 1):
                for j in range(len(ngrams_dict_bsz)):
                    key = tuple(tgt_tokens_list[j][i : i + ngrams_order])
                    if self.pad in key:
                        continue

                    if key in ngrams_dict_bsz[j].keys():
                        ngrams_max_count_bsz[j][ngrams_dict_bsz[j][key]] = (
                            ngrams_max_count_bsz[j][ngrams_dict_bsz[j][key]] + 1
                        )
                    else:
                        ngrams_dict_bsz[j][key] = len(ngrams_list_bsz[j])
                        ngrams_list_bsz[j].append(tgt_tokens_list[j][i : i + ngrams_order])
                        ngrams_max_count_bsz[j].append(1)

            padded_ngrams_num = max([len(ngrams_list) for ngrams_list in ngrams_list_bsz])
            padded_ngrams_template = []
            for i in range(ngrams_order):
                padded_ngrams_template.append(1)

            for i in range(len(ngrams_list_bsz)):
                while len(ngrams_list_bsz[i]) < padded_ngrams_num:
                    ngrams_list_bsz[i].append(padded_ngrams_template)
                    ngrams_max_count_bsz[i].append(0)

            ngrams_tensor_bsz = torch.LongTensor(ngrams_list_bsz).cuda(
                tgt_tokens.device
            )  # bsz, number of ngram, length of ngram
            ngrams_max_count_bsz = torch.tensor(ngrams_max_count_bsz).cuda(tgt_tokens.device)  # bsz, number of ngram
            del ngrams_dict_bsz
            del ngrams_list_bsz

        arrival_prob = torch.ones(transition.size(0), 1).to(transition)
        for i in range(1, transition.size(-1)):
            arrival_prob = torch.cat(
                [arrival_prob, torch.mul(arrival_prob[:, 0:i], transition[:, 0:i, i]).sum(dim=-1).unsqueeze(-1)],
                dim=-1,
            )

        expected_length = arrival_prob.sum(dim=-1)
        expected_tol_num_of_ngrams = arrival_prob.unsqueeze(1)

        for i in range(ngrams_order - 1):
            expected_tol_num_of_ngrams = torch.bmm(expected_tol_num_of_ngrams, transition)

        expected_tol_num_of_ngrams = expected_tol_num_of_ngrams.sum(dim=-1).sum(dim=-1)

        first_word_in_each_gram = ngrams_tensor_bsz[:, :, 0].unsqueeze(-1)  # bsz, number of ngram, 1

        # bsz, number of ngram, prelen
        first_word_probs = torch.gather(
            input=probs.unsqueeze(1).expand(-1, first_word_in_each_gram.size(-2), -1, -1),
            dim=-1,
            index=first_word_in_each_gram.unsqueeze(2).expand(-1, -1, probs.size(-2), -1),
        ).squeeze()

        expected_matched_num_of_ngrams = torch.mul(arrival_prob.unsqueeze(1), first_word_probs)
        del first_word_probs

        for i in range(1, ngrams_order):
            target_at_this_word = ngrams_tensor_bsz[:, :, i].unsqueeze(-1)  # bsz, number of ngram, 1

            # bsz, number of ngram, prelen
            word_probs = torch.gather(
                input=probs.unsqueeze(1).expand(-1, target_at_this_word.size(-2), -1, -1),
                dim=-1,
                index=target_at_this_word.unsqueeze(2).expand(-1, -1, probs.size(-2), -1),
            ).squeeze(dim=-1)

            expected_matched_num_of_ngrams = torch.mul(
                torch.bmm(expected_matched_num_of_ngrams, transition), word_probs
            )
            del word_probs

        expected_matched_num_of_ngrams = expected_matched_num_of_ngrams.sum(dim=-1)

        cutted_expected_matched_num_of_ngrams = torch.min(
            expected_matched_num_of_ngrams, ngrams_max_count_bsz.to(expected_matched_num_of_ngrams)
        ).sum(dim=-1)

        # ngrams_F_score = cutted_expected_matched_num_of_ngrams / (expected_tol_num_of_ngrams[-1] + (tgt_tokens.ne(1).sum(dim=-1) - ngrams_order + 1))
        cutted_precision = cutted_expected_matched_num_of_ngrams / expected_tol_num_of_ngrams
        reverse_length_ratio = tgt_tokens.ne(1).sum(dim=-1) / expected_length
        brief_penalty = torch.min(torch.ones_like(reverse_length_ratio), torch.exp(1.0 - reverse_length_ratio))

        loss = -(brief_penalty * cutted_precision).mean()

        with torch.no_grad():
            length_ratio = (expected_length / tgt_tokens.ne(1).sum(dim=-1)).mean()
            precision = cutted_precision.mean()
            recall = (cutted_expected_matched_num_of_ngrams / (tgt_tokens.ne(1).sum(dim=-1) - ngrams_order + 1)).mean()
            brief_penalty = brief_penalty.mean()

        nsentences, ntokens = tgt_tokens.shape[0], tgt_tokens.ne(self.task.tgt_dict.pad()).sum()
        return {
            "name": name,
            "loss": loss,
            "factor": factor,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "precision": precision,
            "recall": recall,
            "len_ratio": length_ratio,
            "bp": brief_penalty,
        }


class DATransformerDecoder(NATransformerDecoder):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn)
        self.init_link_feature(cfg)

    def init_link_feature(self, cfg):
        links_dim = cfg.decoder.embed_dim
        if cfg.links_position != "none":
            links_dim += cfg.decoder.embed_dim
            self.link_positional = PositionalEmbedding(
                self.max_positions(),
                cfg.decoder.embed_dim,
                self.padding_idx,
                learned=cfg.links_position == "learned",
            )

        self.query_linear = nn.Linear(links_dim, cfg.decoder.embed_dim)
        self.key_linear = nn.Linear(links_dim, cfg.decoder.embed_dim)
        self.gate_linear = nn.Linear(links_dim, cfg.decoder.attention_heads)

    def forward(self, encoder_out, prev_output_tokens, require_links=False, **kwargs):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=False,
        )
        word_ins_out = self.output_layer(features)

        links = self.extract_links(features, prev_output_tokens) if require_links else None

        return word_ins_out, links

    def extract_links(self, features, prev_output_tokens):
        bsz, pre_len, _ = features.size()

        if self.cfg.links_position != "none":
            features = torch.cat([features, self.link_positional(prev_output_tokens)], dim=-1)

        features = features.transpose(0, 1)  # [pre_len, batch_size, hidden_size]

        num_heads = self.cfg.decoder.attention_heads
        head_dim = self.cfg.decoder.embed_dim // self.cfg.decoder.attention_heads

        # use higher precision in training
        target_dtype = torch.float if self.training else self.query_linear.weight.dtype
        logsumexp_fast = logsumexp if self.training else torch.logsumexp

        # Use multiple heads in calculating transition matrix
        q_chunks = (
            self.query_linear(features)
            .contiguous()
            .view(pre_len, bsz * num_heads, head_dim)
            .transpose(0, 1)
            .to(dtype=target_dtype)
        )
        k_chunks = (
            self.key_linear(features)
            .contiguous()
            .view(pre_len, bsz * num_heads, head_dim)
            .transpose(0, 1)
            .to(dtype=target_dtype)
        )

        # The head probability on each position. log_gates: batch_size * pre_len * chunk_num
        log_gates = F.log_softmax(self.gate_linear(features).transpose(0, 1).to(dtype=target_dtype), dim=-1)

        # Transition probability for each head. log_multi_content: batch_size * pre_len * pre_len * chunk_num
        log_multi_content = torch.einsum("bif,bjf->bij", q_chunks, k_chunks) / head_dim**0.5

        # transition_valid_mask specifies all possible transition places for each position
        # transition_valid_mask shape: [batch_size * num_heads, pre_len, pre_len]
        transition_valid_mask = prev_output_tokens.ne(self.pad).unsqueeze(1)

        transition_valid_mask = (
            transition_valid_mask.unsqueeze(1)
            .repeat(1, num_heads, 1, 1)
            .view(bsz * num_heads, *transition_valid_mask.shape[1:])
        )

        # only allows left-to-right transition
        transition_valid_mask = transition_valid_mask & transition_valid_mask.new_ones(pre_len, pre_len).triu_(
            1
        ).unsqueeze(0)

        if self.cfg.max_transition_length != -1:  # finity transition length, prepare for cuda input format
            log_multi_content_extract, link_unused_mask = self.extract_valid_links(
                log_multi_content, transition_valid_mask
            )
            # batch * pre_len * trans_len * chunk_num, batch * pre_len * trans_len
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_unused_mask, float("-inf"))
            log_multi_attention = F.log_softmax(log_multi_content_extract, dim=2)
            log_multi_attention = log_multi_attention.masked_fill(link_unused_mask, float("-inf"))
        else:  # infinity transition length, prepare for torch input format
            link_unused_mask = transition_valid_mask.sum(dim=2) == 0
            transition_valid_mask[link_unused_mask] = True
            log_multi_content = log_multi_content.masked_fill(~transition_valid_mask, float("-inf"))
            log_multi_attention = F.log_softmax(log_multi_content, dim=-1)
            log_multi_attention = log_multi_attention.masked_fill(link_unused_mask.unsqueeze(-1), float("-inf"))

        log_multi_attention = log_multi_attention.view(bsz, num_heads, pre_len, pre_len).permute(0, 2, 3, 1)
        # batch_size * pre_len * pre_len
        links = logsumexp_fast(log_multi_attention + log_gates.unsqueeze(2), dim=-1)

        return links.squeeze(-1)


def da_transformer_base(args):
    pass


@register_model_architecture("dag_transformer", "dag_transformer_pretrain")
def da_transformer_pretrain(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.dropout = getattr(args, "dropout", 0.1)
    da_transformer_base(args)


@register_model_architecture("dag_transformer", "dag_transformer_iwslt_de_en")
def da_transformer_iwslt(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    da_transformer_base(args)


@register_model_architecture("dag_transformer", "dag_transformer_big")
def da_transformer_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    da_transformer_base(args)
