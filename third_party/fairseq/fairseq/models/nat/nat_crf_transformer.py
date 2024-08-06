# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass, field

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.nonautoregressive_transformer import (
    NATransformerConfig,
    NATransformerModel,
)
from fairseq.modules.dynamic_crf_layer import DynamicCRF

logger = logging.getLogger(__name__)


@dataclass
class NACRFTransformerConfig(NATransformerConfig):
    crf_lowrank_approx: int = field(
        default=32, metadata={"help": "the dimension of low-rank approximation of transition"}
    )
    crf_beam_approx: int = field(
        default=64, metadata={"help": "the beam size for apporixmating the normalizing factor"}
    )
    word_ins_loss_factor: float = field(
        default=0.5, metadata={"help": "weights on NAT loss used to co-training with CRF loss."}
    )


@register_model("nacrf_transformer", dataclass=NACRFTransformerConfig)
class NACRFTransformerModel(NATransformerModel):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.crf_layer = DynamicCRF(
            num_embedding=len(self.tgt_dict),
            low_rank=cfg.crf_lowrank_approx,
            beam_size=cfg.crf_beam_approx,
        )

    @property
    def allow_ensemble(self):
        return False

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # decoding
        word_ins_out, _ = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        word_ins_tgt, word_ins_mask = tgt_tokens, tgt_tokens.ne(self.pad)

        # compute the log-likelihood of CRF
        crf_nll = -self.crf_layer(word_ins_out, word_ins_tgt, word_ins_mask)
        crf_nll = crf_nll / word_ins_mask.type_as(crf_nll).sum(-1)

        invalid_loss = crf_nll.isnan() | crf_nll.isinf()
        if invalid_loss.any():
            logger.warning(f"Skipping {invalid_loss.sum()} samples due to NaN or Inf loss.")
            crf_nll = crf_nll[~invalid_loss]
        crf_nll = crf_nll.mean()

        ret = {
            "word_crf": {"loss": crf_nll},
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.cfg.length_loss_factor,
            },
        }
        if self.cfg.word_ins_loss_factor > 0:
            ret["word_ins"] = {
                "out": word_ins_out,
                "tgt": word_ins_tgt,
                "mask": word_ins_mask,
                "nll_loss": True,
                "factor": self.cfg.word_ins_loss_factor,
                "report_accuracy": self.cfg.report_accuracy,
            }

        return ret

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder and get emission scores
        output_masks = output_tokens.ne(self.pad)
        word_ins_out, _ = self.decoder(normalize=False, prev_output_tokens=output_tokens, encoder_out=encoder_out)

        # run viterbi decoding through CRF
        _scores, _tokens = self.crf_layer.forward_decoder(word_ins_out, output_masks)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )


def nacrf_base_architecture(args):
    pass


@register_model_architecture("nacrf_transformer", "nacrf_transformer_wmt_en_de")
def nacrf_transformer_wmt_en_de(args):
    nacrf_base_architecture(args)


@register_model_architecture("nacrf_transformer", "nacrf_transformer_iwslt_de_en")
def nacrf_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    nacrf_base_architecture(args)
