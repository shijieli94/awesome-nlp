# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerDecoderBase,
    TransformerEncoderBase,
)
from omegaconf import II
from torch import Tensor

logger = logging.getLogger(__name__)


QUANT_MODE_CHOICES = ChoiceEnum(["qat", "ptq"])


@register_model("transformer", dataclass=TransformerConfig)
class TransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {'path': path, 'tokenizer': 'moses', 'bpe': 'subword_nmt'}

        def moses_fastbpe(path):
            return {'path': path, 'tokenizer': 'moses', 'bpe': 'fastbpe'}

        def spm(path):
            return {'path': path, 'bpe': 'sentencepiece', 'tokenizer': 'space'}

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
            'transformer.wmt20.en-ta': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-ta.single.tar.gz'),
            'transformer.wmt20.en-iu.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.news.single.tar.gz'),
            'transformer.wmt20.en-iu.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.nh.single.tar.gz'),
            'transformer.wmt20.ta-en': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta-en.single.tar.gz'),
            'transformer.wmt20.iu-en.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.news.single.tar.gz'),
            'transformer.wmt20.iu-en.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.nh.single.tar.gz'),
            'transformer.flores101.mm100.615M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz'),
            'transformer.flores101.mm100.175M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz'),
        }
        # fmt: on

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc(), delete_default=True, with_prefix="")

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        cfg = TransformerConfig.from_namespace(cfg)

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError("--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim")
            if cfg.decoder.embed_path and (cfg.decoder.embed_path != cfg.encoder.embed_path):
                raise ValueError("--share-all-embeddings not compatible with --decoder-embed-path")
            encoder_embed_tokens = cls.build_embedding(cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path)
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(cfg, src_dict, cfg.encoder.embed_dim)
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path)
            decoder_embed_tokens = cls.build_embedding(cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path)
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = False,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class TransformerModel(TransformerModelBase):
    pass


@dataclass
class LSTransformerConfig(TransformerConfig):
    use_torch_layer: bool = field(
        default=False, metadata={"help": "use custom torch layer instead of LightSeq cuda layer"}
    )
    enable_quant: bool = field(default=False, metadata={"help": "enable quantization"})
    quant_mode: str = field(default="qat", metadata={"help": "quantization mode"})
    # args for Gradient Communication Quantization (GCQ) in multi-machine distributed training
    enable_GCQ: bool = field(default=False, metadata={"help": "enable gradient communication quantization"})
    GCQ_quantile: float = field(
        default=0.99, metadata={"help": "quantile value of gradient communication quantization, between 0.0-1.0"}
    )
    max_tokens: Optional[int] = II("dataset.max_tokens")
    device_id: int = II("distributed_training.device_id")


@register_model("ls_transformer", dataclass=LSTransformerConfig)
class LSTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg

    @classmethod
    def build_model(cls, cfg, task):
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder_embed_dim != cfg.decoder_embed_dim:
                raise ValueError("--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim")
            encoder_embed_tokens = cls.build_embedding(cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.max_positions)
            decoder_embed_tokens = cls.build_embedding(
                cfg,
                tgt_dict,
                cfg.decoder.embed_dim,
                cfg.decoder.max_positions,
                emb_lookup=encoder_embed_tokens,
            )
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.max_positions)
            decoder_embed_tokens = cls.build_embedding(cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.max_positions)

        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)

        from lightseq.training.ops.pytorch.quantization import (
            disable_quant,
            enable_quant,
            ptq_mode,
            qat_mode,
        )

        if cfg.enable_quant:
            if cfg.use_torch_layer:
                if cfg.quant_mode == "qat":
                    encoder.apply(qat_mode)
                    decoder.apply(qat_mode)
                elif cfg.quant_mode == "ptq":
                    encoder.apply(ptq_mode)
                    decoder.apply(ptq_mode)
            else:
                if cfg.quant_mode == "qat":
                    encoder.apply(enable_quant)
                    decoder.apply(enable_quant)
                    encoder.apply(qat_mode)
                    decoder.apply(qat_mode)

                else:
                    raise NotImplementedError
        else:
            encoder.apply(disable_quant)
            decoder.apply(disable_quant)

        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, max_positions, emb_lookup=None, **kwargs):
        from lightseq.training.ops.pytorch.layer_base import (
            TransformerEmbeddingLayerBase,
        )

        use_torch_layer = cfg.use_torch_layer or cfg.no_scale_embedding or cfg.layernorm_embedding
        config = TransformerEmbeddingLayerBase.get_config(
            vocab_size=len(dictionary),
            embedding_dim=embed_dim,
            max_batch_tokens=cfg.max_tokens,
            max_seq_len=max_positions,
            padding_idx=dictionary.pad(),
            dropout=cfg.dropout,
            fp16=cfg.fp16,
            local_rank=cfg.device_id,
            trainable_pos=(cfg.encoder.learned_pos or cfg.decoder.learned_pos),
            no_scale_embedding=cfg.no_scale_embedding,
            layernorm_embedding=cfg.layernorm_embedding,
            need_offset=("bart" in cfg.arch),
        )
        if use_torch_layer:
            from lightseq.training.ops.pytorch.torch_transformer_layers import (
                TransformerEmbeddingLayer,
            )

            if emb_lookup is not None:
                emb_lookup = emb_lookup.emb_lookup
            emb = TransformerEmbeddingLayer(config, emb_lookup=emb_lookup)
        else:
            from lightseq.training.ops.pytorch.transformer_embedding_layer import (
                LSTransformerEmbeddingLayer as TransformerEmbeddingLayer,
            )

            if emb_lookup is not None:
                return emb_lookup
            emb = TransformerEmbeddingLayer(config)

        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        from .transformer_encoder import LSTransformerEncoder

        return LSTransformerEncoder(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        from .transformer_decoder import LSTransformerDecoder

        return LSTransformerDecoder(cfg, tgt_dict, embed_tokens)

    def forward(self, src_tokens, prev_output_tokens, features_only=False, **kwargs):
        encoder_out = self.encoder(src_tokens)
        decoder_out = self.decoder(prev_output_tokens, encoder_out, features_only=features_only)
        return decoder_out


@register_model_architecture("transformer", "transformer_tiny")
@register_model_architecture("ls_transformer", "ls_transformer_tiny")
def tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    base_architecture(args)


def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.merge_src_tgt_embed = getattr(args, "merge_src_tgt_embed", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("transformer", "transformer_iwslt_de_en")
@register_model_architecture("ls_transformer", "ls_transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_wmt_en_de")
@register_model_architecture("ls_transformer", "ls_transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer", "transformer_vaswani_wmt_en_de_big")
@register_model_architecture("ls_transformer", "ls_transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_vaswani_wmt_en_fr_big")
@register_model_architecture("ls_transformer", "ls_transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer", "transformer_wmt_en_de_big")
@register_model_architecture("ls_transformer", "ls_transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer", "transformer_wmt_en_de_big_t2t")
@register_model_architecture("ls_transformer", "ls_transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)
