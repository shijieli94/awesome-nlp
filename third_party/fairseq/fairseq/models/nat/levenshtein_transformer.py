# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.fairseq_nat_model import (
    FairseqNATDecoder,
    FairseqNATModel,
    ensemble_decoder,
)
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_layer import TransformerDecoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .levenshtein_utils import (
    _apply_del_words,
    _apply_ins_masks,
    _apply_ins_words,
    _fill,
    _get_del_targets,
    _get_ins_targets,
    _skip,
    _skip_encoder_out,
)
from .nonautoregressive_transformer import NATransformerConfig, NATransformerDecoder


@dataclass
class LevenshteinTransformerModelConfig(NATransformerConfig):
    early_exit: List[int] = field(
        default_factory=lambda: [6, 6, 6],
        metadata={"help": "number of decoder layers before word_del, mask_ins, word_ins"},
    )
    no_share_discriminator: bool = field(default=False, metadata={"help": "separate parameters for discriminator"})
    no_share_maskpredictor: bool = field(default=False, metadata={"help": "separate parameters for mask-predictor"})
    share_discriminator_maskpredictor: bool = field(
        default=False, metadata={"help": "share the parameters for both mask-predictor and discriminator"}
    )
    sampling_for_deletion: bool = field(
        default=False, metadata={"help": "instead of argmax, use sampling to predict the tokens"}
    )


@register_model("levenshtein_transformer", dataclass=LevenshteinTransformerModelConfig)
class LevenshteinTransformerModel(FairseqNATModel):
    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = LevenshteinTransformerDecoder(cfg, tgt_dict, embed_tokens)
        if cfg.apply_bert_init:
            decoder.apply(init_bert_params)
        return decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        assert tgt_tokens is not None, "forward function only supports training."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            prev_output_tokens, tgt_tokens, self.pad, self.unk
        )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
        mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            prev_output_tokens=masked_tgt_tokens,
            encoder_out=encoder_out,
        )

        # make online prediction
        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1).view(
                word_ins_out.size(0), -1
            )
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(~masked_tgt_masks, tgt_tokens[~masked_tgt_masks])

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
        word_del_out, _ = self.decoder.forward_word_del(
            normalize=False,
            prev_output_tokens=word_predictions,
            encoder_out=encoder_out,
        )
        word_del_masks = word_predictions.ne(self.pad)

        return {
            "mask_ins": {
                "out": mask_ins_out,
                "tgt": mask_ins_targets,
                "mask": mask_ins_masks,
                "ls": 0.01,
            },
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": masked_tgt_masks,
                "nll_loss": True,
            },
            "word_del": {
                "out": word_del_out,
                "tgt": word_del_targets,
                "mask": word_del_masks,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history

        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if not encoder_out["encoder_padding_mask"]:
                max_src_len = encoder_out["encoder_out"].size(0)
                src_lens = encoder_out["encoder_out"].new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out["encoder_padding_mask"][0]).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            word_del_score, word_del_attn = self.decoder.forward_word_del(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_del_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_del_word),
            )
            word_del_pred = word_del_score.max(-1)[1].bool()
            word_del_attn = word_del_attn["cross_attns"][-1] if len(word_del_attn["cross_attns"]) > 0 else None

            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                word_del_attn,
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            attn = _fill(attn, can_del_word, _attn, 0.0)

            if history is not None:
                history.append({"tokens": output_tokens.clone()})

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_mask),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_mask),
            )
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred))

            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                history.append({"tokens": output_tokens.clone()})

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_word),
            )
            word_ins_score, word_ins_pred = word_ins_score.max(-1)
            word_ins_attn = word_ins_attn["cross_attns"][-1] if len(word_ins_attn["cross_attns"]) > 0 else None
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.0)

            if history is not None:
                history.append({"tokens": output_tokens.clone()})

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, **kwargs):
        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens[:, 1] = self.eos

        initial_output_scores = initial_output_tokens.new_zeros(*initial_output_tokens.size()).type_as(
            encoder_out["encoder_out"][0]
        )

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=1,
            max_step=1,
            history=None,
        )


class LevenshteinTransformerDecoder(NATransformerDecoder):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.sampling_for_deletion = getattr(cfg, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in cfg.early_exit]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if cfg.no_share_maskpredictor:
            self.layers_msk = nn.ModuleList(
                [TransformerDecoderLayer(cfg, no_encoder_attn) for _ in range(self.early_exit[1])]
            )
        self.layers_del = None
        if cfg.no_share_discriminator:
            self.layers_del = nn.ModuleList(
                [TransformerDecoderLayer(cfg, no_encoder_attn) for _ in range(self.early_exit[0])]
            )

        if cfg.share_discriminator_maskpredictor:
            assert getattr(cfg, "no_share_discriminator", False), "must set saperate discriminator"
            self.layers_msk = self.layers_del

    @ensemble_decoder
    def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[1],
            layers=self.layers_msk,
            **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra
        return decoder_out, extra

    @ensemble_decoder
    def forward_word_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[2], layers=self.layers, **unused
        )
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra
        return decoder_out, extra

    @ensemble_decoder
    def forward_word_del(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[0],
            layers=self.layers_del,
            **unused
        )
        decoder_out = F.linear(features, self.embed_word_del.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra
        return decoder_out, extra


def levenshtein_base_architecture(args):
    pass


@register_model_architecture("levenshtein_transformer", "levenshtein_transformer_wmt_en_de")
def levenshtein_transformer_wmt_en_de(args):
    levenshtein_base_architecture(args)


# similar parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("levenshtein_transformer", "levenshtein_transformer_vaswani_wmt_en_de_big")
def levenshtein_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    levenshtein_base_architecture(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("levenshtein_transformer", "levenshtein_transformer_wmt_en_de_big")
def levenshtein_transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    levenshtein_transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("levenshtein_transformer", "levenshtein_transformer_iwslt_de_en")
def levenshtein_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    levenshtein_base_architecture(args)
