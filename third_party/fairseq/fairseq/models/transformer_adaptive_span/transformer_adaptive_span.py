# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

from .adaptive_span_model import TransformerSeq as AdaptiveSpanTransformerModel

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSpanSmallConfig(FairseqDataclass):
    # defaults come from https://github.com/facebookresearch/adaptive-span/blob/master/experiments/enwik8_small.sh
    cutoffs: List[int] = field(default_factory=lambda: [20000, 40000, 200000])
    div_val: int = field(default=1, metadata={"help": "divident value for adapative input and softmax"})
    d_model: int = field(default=256, metadata={"help": "model dimension"})
    n_heads: int = field(default=4, metadata={"help": "number of heads"})
    d_inner: int = field(default=1024, metadata={"help": "inner dimension in FF"})
    n_layers: int = field(default=8, metadata={"help": "number of total layers"})
    attn_span: int = 1024
    dropout: float = field(default=0.0, metadata={"help": "global dropout rate"})
    emb_dropout: float = 0.0
    adapt_span_ramp: int = 32
    adapt_span_init: float = 0.0
    aux_loss_scaler: float = 0.000002
    adapt_span_layer: bool = False


@register_model("transformer_adaptive_span", dataclass=AdaptiveSpanSmallConfig)
class AdaptiveSpanTransformer(FairseqLanguageModel):
    @classmethod
    def build_model(cls, cfg: AdaptiveSpanSmallConfig, task):
        return cls(AdaptiveSpanDecoder(cfg, task))

    def get_aux_loss(self):
        return self.decoder.get_aux_loss()

    def get_current_max_span(self):
        return self.decoder.get_current_max_span()

    def get_current_avg_span(self):
        return self.decoder.get_current_avg_span()


class AdaptiveSpanDecoder(FairseqIncrementalDecoder):
    def __init__(self, cfg, task):
        self.cfg = cfg
        super().__init__(task.target_dictionary)
        kwargs = dict(self.cfg)
        self.model = AdaptiveSpanTransformerModel(
            vocab_size=len(task.target_dictionary),
            adapt_io_params={
                "adapt_io_enabled": False,
                "adapt_io_cutoffs": cfg.cutoffs,
                "adapt_io_divval": cfg.div_val,
                "adapt_io_tied": True,
            },
            pers_mem_params={"pers_mem_size": 2048},
            adapt_span_params={
                "adapt_span_enabled": True,
                "adapt_span_loss": kwargs.pop("aux_loss_scaler"),
                "adapt_span_ramp": kwargs.pop("adapt_span_ramp"),
                "adapt_span_init": kwargs.pop("adapt_span_init"),
                "adapt_span_layer": kwargs.pop("adapt_span_layer"),
                "adapt_span_cache": True,
            },
            **kwargs,
        )
        self._mems = None

    def forward(
        self,
        src_tokens,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
        **kwargs,
    ):
        bsz = src_tokens.size(0)
        if incremental_state is not None:  # used during inference
            mems = self.get_incremental_state("mems")
            src_tokens = src_tokens[:, -1:]  # only keep the most recent token
        else:
            mems = self._mems

        if mems is None:
            # first time init
            mems = self.init_hid_cache(bsz)
        output = self.model(x=src_tokens, h_cache=mems)
        if incremental_state is not None:
            self.set_incremental_state(incremental_state, "mems", output[1])
        else:
            self._mems = output[1]
        return (output[0],)

    def max_positions(self):
        return self.cfg.attn_span

    def init_hid_cache(self, batch_sz):
        hid = []
        for layer in self.model.layers:
            param = next(self.model.parameters())
            h = torch.zeros(
                batch_sz,
                layer.get_cache_size(),
                self.cfg.d_model,
                dtype=param.dtype,
                device=param.device,
            )
            hid.append(h)
        return hid

    def get_aux_loss(self):
        return self.model.get_aux_loss()

    def get_current_max_span(self):
        return self.model.get_current_max_span()

    def get_current_avg_span(self):
        return self.model.get_current_avg_span()

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        new_order: torch.Tensor,
    ):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        raise NotImplementedError("This is required for generation/beam search")
        # mems = self.get_incremental_state(incremental_state, "mems")
        # if mems is not None:
        #     new_mems = [mems_i.index_select(1, new_order) for mems_i in mems]
        #     self.set_incremental_state(incremental_state, "mems", new_mems)


@register_model_architecture("transformer_adaptive_span", "transformer_adaptive_span_enwik8")
def transformer_adaptive_span_enwik8(args):
    args.n_layers = getattr(args, "n_layers", 12)
    args.d_model = getattr(args, "d_model", 512)
    args.d_inner = getattr(args, "d_inner", 2048)
    args.n_heads = getattr(args, "n_heads", 8)
    args.attn_span = getattr(args, "attn_span", 8192)
    args.dropout = getattr(args, "dropout", 0.3)
    args.aux_loss_scaler = getattr(args, "aux_loss_scaler", 5e-7)


@register_model_architecture("transformer_adaptive_span", "transformer_adaptive_span_enwik8_large")
def transformer_adaptive_span_enwik8(args):
    args.n_layers = getattr(args, "n_layers", 24)
    args.d_model = getattr(args, "d_model", 768)
    args.d_inner = getattr(args, "d_inner", 4096)
    args.n_head = getattr(args, "n_head", 8)
    args.attn_span = getattr(args, "attn_span", 8192)
    args.dropout = getattr(args, "dropout", 0.4)
    args.aux_loss_scaler = getattr(args, "aux_loss_scaler", 5e-7)


@register_model_architecture("transformer_adaptive_span", "transformer_adaptive_span_wt103")
def transformer_adaptive_span_enwik8(args):
    args.n_layers = getattr(args, "n_layers", 36)
    args.d_model = getattr(args, "d_model", 512)
    args.d_inner = getattr(args, "d_inner", 4096)
    args.n_head = getattr(args, "n_head", 8)
    args.attn_span = getattr(args, "attn_span", 2048)
    args.dropout = getattr(args, "dropout", 0.4)
    args.aux_loss_scaler = getattr(args, "aux_loss_scaler", 5e-7)
