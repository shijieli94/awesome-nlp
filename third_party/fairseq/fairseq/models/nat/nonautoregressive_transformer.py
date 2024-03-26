# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.dataclass import ChoiceEnum
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.fairseq_nat_model import (
    FairseqNATDecoder,
    FairseqNATEncoder,
    FairseqNATModel,
    ensemble_decoder,
    ensemble_encoder,
)
from fairseq.models.transformer import Embedding, TransformerConfig
from fairseq.modules.transformer_sentence_encoder import init_bert_params

LENGTH_PRED_METHOD_CHOICES = ChoiceEnum(["mean", "bert", "none"])


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = ((enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


@dataclass
class NATransformerConfig(TransformerConfig):
    apply_bert_init: bool = field(default=False, metadata={"help": "use custom param initialization for BERT"})
    # length prediction
    src_embedding_copy: bool = field(
        default=False, metadata={"help": "copy encoder word embeddings as the initial input of the decoder"}
    )
    pred_length_offset: bool = field(
        default=False, metadata={"help": "predicting the length difference between the target and source sentences"}
    )
    sg_length_pred: bool = field(
        default=False, metadata={"help": "stop the gradients back-propagated from the length predictor"}
    )
    length_loss_factor: float = field(default=0.1, metadata={"help": "weights on the length prediction loss"})
    length_pred_method: LENGTH_PRED_METHOD_CHOICES = field(
        default="mean", metadata={"help": "predict the length with either the average length or a prediction network"}
    )
    report_accuracy: bool = field(default=False, metadata={"help": "report accuracy metric"})


@register_model("nonautoregressive_transformer", dataclass=NATransformerConfig)
class NATransformerModel(FairseqNATModel):
    @property
    def allow_length_beam(self):
        return True

    def nonspecial_mask(self, token):
        return token.ne(self.pad) & token.ne(self.eos) & token.ne(self.bos)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(cfg, tgt_dict, embed_tokens)
        if cfg.apply_bert_init:
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        encoder = NATransformerEncoder(cfg, src_dict, embed_tokens)
        if cfg.apply_bert_init:
            encoder.apply(init_bert_params)
        return encoder

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
            embedding_copy=self.cfg.src_embedding_copy,
        )

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "nll_loss": True,
                "report_accuracy": self.cfg.report_accuracy,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.cfg.length_loss_factor,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        model_output, _ = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            embedding_copy=self.cfg.src_embedding_copy,
        )
        _scores, _tokens = model_output.max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append({"tokens": output_tokens.clone(), "scores": output_scores.clone()})

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, beam_size=1, length_format=None):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
            beam_size=beam_size,
        )
        if beam_size > 1 and length_format != "topk":
            length_tgt = length_tgt[:, [0]] + utils.new_arange(length_tgt, 1, beam_size) - beam_size // 2
        length_tgt = length_tgt.view(-1)

        return self._initialize_output_tokens_with_length(encoder_out, src_tokens, length_tgt)

    def _initialize_output_tokens_with_length(self, encoder_out, src_tokens, length_tgt):
        length_tgt = length_tgt.clamp_(min=2, max=self.max_decoder_positions())
        max_length = length_tgt.max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(length_tgt.size(0), max_length).fill_(self.pad)
        initial_output_tokens.masked_fill_(idx_length[None, :] < length_tgt[:, None], self.unk)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

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


class NATransformerEncoder(FairseqNATEncoder):
    def __init__(self, cfg, dictionary, embed_tokens):
        super().__init__(cfg, dictionary, embed_tokens)
        if cfg.length_pred_method == "bert":
            self.length_token = Embedding(1, embed_tokens.embedding_dim, None)
        else:
            self.length_token = None

    @ensemble_encoder
    def forward(self, src_tokens, src_lengths=None, return_fc=False, return_all_hiddens=False, token_embeddings=None):
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

        # encoder layers
        for idx, layer in enumerate(self.layers):
            x, layer_attn, fc_result = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                return_fc=return_fc,
                need_attn=self.cfg.encoder.return_attns,
                need_head_weights=self.cfg.encoder.return_head_attns,
            )

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x if self.length_token is None else x[1:])
                if return_fc:
                    fc_results.append(fc_result)
            if self.cfg.encoder.return_attns:
                attns.append(layer_attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
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
        }


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()

        if cfg.length_pred_method != "none":
            self.embed_length = Embedding(
                256 if cfg.pred_length_offset else self.max_positions(), cfg.encoder.embed_dim, None
            )

    @ensemble_decoder
    def forward(
        self, normalize, encoder_out, prev_output_tokens, return_all_hiddens=False, embedding_copy=False, **unused
    ):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=embedding_copy,
            return_all_hiddens=return_all_hiddens,
            **unused,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out, extra

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        if self.cfg.length_pred_method == "mean":
            enc_feats = encoder_out["encoder_out"][0]  # T x B x C
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
            else:
                src_masks = None
            enc_feats = _mean_pooling(enc_feats, src_masks)
        elif self.cfg.length_pred_method == "bert":
            enc_feats = encoder_out["encoder_out"][1]

        if self.cfg.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        return_all_hiddens=False,
        layers=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
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

            x, attn, cross_attn, _ = layer(
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
            )
            if return_all_hiddens:
                inner_states.append(x)
            if self.cfg.decoder.return_attns:
                attns.append(attn)
                cross_attns.append(cross_attn)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attns": attns, "cross_attns": cross_attns, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None, return_pe=False):
        # embed positions
        positions = self.embed_positions(prev_output_tokens) if self.embed_positions is not None else None

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.quant_noise is not None:
                x = self.quant_noise(x)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        if return_pe:
            return x, decoder_padding_mask, positions
        return x, decoder_padding_mask

    def forward_copying_source(self, encoder_out, prev_output_tokens):
        src_embed = encoder_out["encoder_embedding"][0]
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_mask = encoder_out["encoder_padding_mask"][0]
        else:
            src_mask = None
        src_mask = ~src_mask if src_mask is not None else prev_output_tokens.new_ones(*src_embed.size()[:2]).bool()
        tgt_mask = prev_output_tokens.ne(self.padding_idx)

        length_sources = src_mask.sum(1)
        length_targets = tgt_mask.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets)
        if not all(src_mask[:, 0]):  # left-padded
            mapped_inputs += (length_sources.max() - length_sources)[:, None]
        mapped_inputs = mapped_inputs.masked_fill(~tgt_mask, 0)
        copied_embedding = torch.gather(
            src_embed,
            1,
            mapped_inputs.unsqueeze(-1).expand(*mapped_inputs.size(), src_embed.size(-1)),
        )
        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None, beam_size=1):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.cfg.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(enc_feats.size(0))
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
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


def base_architecture(args):
    pass


@register_model_architecture("nonautoregressive_transformer", "nonautoregressive_transformer_wmt_en_de")
def nonautoregressive_transformer_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("nonautoregressive_transformer", "nonautoregressive_transformer_iwslt_de_en")
def nonautoregressive_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)
