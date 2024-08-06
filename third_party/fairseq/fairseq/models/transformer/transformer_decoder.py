# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import transformer_layer
from fairseq.modules.adaptive_softmax import AdaptiveSoftmax
from fairseq.modules.base_layer import BaseLayer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.layer_drop import LayerDropModuleList
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.positional_embedding import PositionalEmbedding
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


class TransformerDecoderBase(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        cfg (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__))
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = getattr(cfg, "max_target_positions", None) or cfg.decoder.max_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_decoder_layer(cfg, no_encoder_attn, layer_idx=i) for i in range(cfg.decoder.layers)]
        )
        if cfg.decoder.layers_to_share:
            layers_to_share = list(map(int, cfg.decoder.layers_to_share.split(",")))
            assert set(layers_to_share) == set(
                range(len(self.layers))
            ), "Missing or unexpected index found in `--decoder-layers-to-share`"
            self.layers = [self.layers[idx] for idx in layers_to_share]

        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(self.output_embed_dim, len(dictionary), bias=False)
            nn.init.normal_(self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5)
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def build_decoder_layer(self, cfg, no_encoder_attn=False, layer=None, layer_idx=None):
        if layer is None:
            layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        **kwargs,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            return_all_hiddens=return_all_hiddens,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        return_all_hiddens: bool = False,
        **kwargs,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            return_all_hiddens,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(prev_output_tokens, incremental_state=incremental_state)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attns: List[Optional[Tensor]] = []
        cross_attns: List[Optional[Tensor]] = []
        inner_states: List[Optional[Tensor]] = []
        if return_all_hiddens:
            inner_states.append(x)

        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, layer_cross_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=self.cfg.decoder.return_attns,
                need_head_weights=self.cfg.decoder.return_head_attns,
            )
            if return_all_hiddens:
                inner_states.append(x)
            if self.cfg.decoder.return_attns:
                attns.append(layer_attn)
                cross_attns.append(layer_cross_attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attns": attns, "cross_attns": cross_attns, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[embed_out_key]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict["{}.layers.{}.{}.{}".format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(TransformerConfig.from_namespace(args), dictionary, embed_tokens)

    def build_decoder_layer(self, args, no_encoder_attn=False, layer=None):
        return super().build_decoder_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn, layer=layer
        )


class LSTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, cfg, dictionary, embed_tokens, **kwargs):
        self.cfg = cfg
        super().__init__(dictionary)

        self._future_mask = torch.empty(0)

        embed_dim = cfg.decoder.embed_dim
        self.embed_tokens = embed_tokens
        self.padding_idx = self.embed_tokens.config.padding_idx

        self.layers = nn.ModuleList([self.build_decoder_layer(cfg) for _ in range(cfg.decoder.layers)])
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        if cfg.use_torch_layer:
            from lightseq.training.ops.pytorch.quantization import QuantLinear

            self.output_projection = QuantLinear(
                self.embed_tokens.embeddings.shape[1],
                self.embed_tokens.embeddings.shape[0],
                bias=False,
            )
            self.output_projection.weight_quant = self.embed_tokens.emb_quant
            self.output_projection.weight = self.embed_tokens.embeddings
        else:
            from lightseq.training.ops.pytorch.quant_linear_layer import (
                LSQuantLinearLayer,
            )

            config = LSQuantLinearLayer.get_config(
                max_batch_tokens=self.cfg.max_tokens,
                in_features=self.embed_tokens.config.embedding_dim,
                out_features=self.embed_tokens.config.vocab_size,
                bias=False,
                fp16=self.cfg.fp16,
                local_rank=self.cfg.device_id,
            )
            self.output_projection = LSQuantLinearLayer(config)
            del self.output_projection.weight

        self.quant_mode = cfg.enable_quant
        self.use_torch_layer = cfg.use_torch_layer

    def build_decoder_layer(self, cfg):
        if cfg.use_torch_layer:
            from lightseq.training.ops.pytorch.torch_transformer_layers import (
                TransformerDecoderLayer,
            )
        else:
            from fairseq.modules.ls_transformer_decoder_layer import (
                LSFSTransformerDecoderLayer as TransformerDecoderLayer,
            )

        config = TransformerDecoderLayer.get_config(
            max_batch_tokens=cfg.max_tokens,
            max_seq_len=cfg.decoder.max_positions,
            hidden_size=cfg.decoder.embed_dim,
            intermediate_size=cfg.decoder.ffn_embed_dim,
            nhead=cfg.decoder.attention_heads,
            attn_prob_dropout_ratio=cfg.attention_dropout,
            activation_dropout_ratio=cfg.activation_dropout,
            hidden_dropout_ratio=cfg.dropout,
            pre_layer_norm=cfg.decoder.normalize_before,
            fp16=cfg.fp16,
            local_rank=cfg.device_id,
            nlayer=cfg.decoder.layers,
            activation_fn=cfg.activation_fn,
        )
        return TransformerDecoderLayer(config)

    def forward_embedding(self, prev_output_tokens, incremental_state=None):
        step = 0
        if incremental_state is not None:
            step = prev_output_tokens.size(1) - 1
            prev_output_tokens = prev_output_tokens[:, -1:]

        x = self.embed_tokens(prev_output_tokens, step)
        return x, prev_output_tokens

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, features_only=False, **kwargs):
        x, prev_output_tokens = self.forward_embedding(prev_output_tokens, incremental_state)

        if not self.use_torch_layer:
            self.output_projection.weight = self.embed_tokens.para[
                : self.embed_tokens.config.vocab_size * self.embed_tokens.config.embedding_dim
            ].reshape(
                self.embed_tokens.config.vocab_size,
                self.embed_tokens.config.embedding_dim,
            )
            if self.quant_mode:
                self.output_projection.clip_max[1] = self.embed_tokens.para[-1].data

        # x: [batch_size, seq_len, hidden_size]
        for _, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, _, _ = layer(
                x,
                encoder_out=encoder_out.encoder_out,
                encoder_padding_mask=encoder_out.encoder_padding_mask,
                self_attn_mask=self_attn_mask,
                incremental_state=incremental_state,
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only:
            x = self.output_projection(x)
        return x, None

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.cfg.decoder.max_positions

    def buffered_future_mask(self, tensor):
        tensor = tensor.transpose(0, 1)
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]
