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
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from omegaconf import II

from .configuration_transfo_xl import TransfoXLConfig
from .modeling_transfo_xl import TransfoXLLMHeadModel

logger = logging.getLogger(__name__)


@dataclass
class TransformerXLConfig(FairseqDataclass):
    # defaults come from the original Transformer-XL code
    cutoffs: List[int] = field(default_factory=lambda: [20000, 40000, 200000])
    n_layer: int = field(default=12, metadata={"help": "number of total layers"})
    n_head: int = field(default=10, metadata={"help": "number of heads"})
    d_head: int = field(default=50, metadata={"help": "head dimension"})
    d_model: int = field(default=500, metadata={"help": "model dimension"})
    d_inner: int = field(default=1000, metadata={"help": "inner dimension in FF"})
    dropout: float = field(default=0.0, metadata={"help": "global dropout rate"})
    dropatt: float = field(default=0.0, metadata={"help": "attention probability dropout rate"})
    mem_len: int = field(
        default=II("task.tokens_per_sample"), metadata={"help": "length of the retained previous heads"}
    )
    div_val: int = field(default=1, metadata={"help": "divident value for adapative input and softmax"})
    clamp_len: int = field(default=-1, metadata={"help": "use the same pos embeddings after clamp_len"})
    same_length: bool = field(default=False, metadata={"help": "use the same attn length for all tokens"})
    attn_type: int = field(
        default=0,
        metadata={
            "help": "attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al."
        },
    )
    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute"
        },
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations."},
    )


@register_model("transformer_xl", dataclass=TransformerXLConfig)
class TransformerXLLanguageModel(FairseqLanguageModel):
    @classmethod
    def build_model(cls, cfg: TransformerXLConfig, task):
        return cls(TransformerXLDecoder(cfg, task))


class TransformerXLDecoder(FairseqIncrementalDecoder):
    def __init__(self, cfg, task):
        super().__init__(task.target_dictionary)
        self.cfg = cfg

        if cfg.attn_type > 0:
            logger.warning(
                "attn_type != 0 is not supported in pretrained transformer-xl model. You may need to train it from scratch."
            )

        # remove any cutoffs larger than the vocab size
        cutoffs = [cutoff for cutoff in cfg.cutoffs if cutoff < len(task.target_dictionary)]

        config = TransfoXLConfig(
            vocab_size=len(task.target_dictionary),
            cutoffs=cutoffs,
            d_model=cfg.d_model,
            d_embed=cfg.d_model,
            n_head=cfg.n_head,
            d_head=cfg.d_head,
            d_inner=cfg.d_inner,
            div_val=cfg.div_val,
            n_layer=cfg.n_layer,
            mem_len=cfg.mem_len,
            clamp_len=cfg.clamp_len,
            same_length=cfg.same_length,
            dropout=cfg.dropout,
            dropatt=cfg.dropatt,
            attn_type=cfg.attn_type,
        )
        logger.info(config)
        self.model = TransfoXLLMHeadModel(config)

        if cfg.checkpoint_activations or cfg.offload_activations:
            for i in range(len(self.model.transformer.layers)):
                self.model.transformer.layers[i] = checkpoint_wrapper(
                    self.model.transformer.layers[i],
                    offload_to_cpu=cfg.offload_activations,
                )
                # TODO: may save mem to wrap(layer.pos_ff.CoreNet[3])

        self._mems = None

    def forward(
        self,
        src_tokens,
        src_lengths=None,  # unused
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
    ):
        if incremental_state is not None:  # used during inference
            mems = self.get_incremental_state(incremental_state, "mems")
            src_tokens = src_tokens[:, -1:]  # only keep the most recent token
        else:
            mems = self._mems

        output = self.model(input_ids=src_tokens, mems=mems, return_dict=False)

        if len(output) >= 2:
            if incremental_state is not None:
                self.set_incremental_state(incremental_state, "mems", output[1])
            else:
                self._mems = output[1]

        return (output[0],)

    def max_positions(self):
        return None

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
        mems = self.get_incremental_state(incremental_state, "mems")
        if mems is not None:
            new_mems = [mems_i.index_select(1, new_order) for mems_i in mems]
            self.set_incremental_state(incremental_state, "mems", new_mems)


@register_model_architecture("transformer_xl", "transformer_xl_wt103_base")
def transformer_xl_wt103_base(args):
    args.n_layer = getattr(args, "n_layer", 16)
    args.d_model = getattr(args, "d_model", 410)
    args.n_head = getattr(args, "n_head", 10)
    args.d_head = getattr(args, "d_head", 41)
    args.d_inner = getattr(args, "d_inner", 2100)
    args.dropout = getattr(args, "dropout", 0.1)
    args.dropatt = getattr(args, "dropatt", 0.0)
    args.mem_len = getattr(args, "mem_len", 150)


@register_model_architecture("transformer_xl", "transformer_xl_wt103_large")
def transformer_xl_wt103_large(args):
    args.div_val = getattr(args, "div_val", 4)
    args.n_layer = getattr(args, "n_layer", 18)
    args.d_model = getattr(args, "d_model", 1024)
    args.n_head = getattr(args, "n_head", 16)
    args.d_head = getattr(args, "d_head", 64)
    args.d_inner = getattr(args, "d_inner", 4096)
    args.dropout = getattr(args, "dropout", 0.2)
    args.dropatt = getattr(args, "dropatt", 0.2)


@register_model_architecture("transformer_xl", "transformer_xl_text8_base")
def transformer_xl_text8_base(args):
    args.n_layer = getattr(args, "n_layer", 12)
    args.d_model = getattr(args, "d_model", 512)
    args.n_head = getattr(args, "n_head", 8)
    args.n_head = getattr(args, "n_head", 64)
    args.d_inner = getattr(args, "d_inner", 2048)
    args.dropout = getattr(args, "dropout", 0.1)
    args.dropatt = getattr(args, "dropatt", 0.0)
