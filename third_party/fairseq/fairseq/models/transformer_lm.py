# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Optional

from fairseq import options
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    Embedding,
    TransformerConfig,
    TransformerDecoderBase,
)
from fairseq.modules.adaptive_input import AdaptiveInput
from fairseq.modules.character_token_embedder import CharacterTokenEmbedder
from fairseq.utils import safe_getattr, safe_hasattr
from omegaconf import II


@dataclass
class TransformerLanguageModelConfig(TransformerConfig):
    character_embeddings: bool = field(
        default=False,
        metadata={"help": "if set, uses character embedding convolutions to produce token embeddings"},
    )
    character_filters: str = field(
        default="[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
        metadata={"help": "size of character embeddings"},
    )
    character_embedding_dim: int = field(default=4, metadata={"help": "size of character embeddings"})
    char_embedder_highway_layers: int = field(
        default=2,
        metadata={"help": "number of highway layers for character token embeddder"},
    )

    # NormFormer
    scale_fc: Optional[bool] = field(
        default=False,
        metadata={"help": "Insert LayerNorm between fully connected layers"},
    )
    scale_attn: Optional[bool] = field(default=False, metadata={"help": "Insert LayerNorm after attention"})
    scale_heads: Optional[bool] = field(
        default=False,
        metadata={"help": "Learn a scale coefficient for each attention head"},
    )
    scale_resids: Optional[bool] = field(
        default=False,
        metadata={"help": "Learn a scale coefficient for each residual connection"},
    )
    # options from other parts of the config
    tokens_per_sample: int = II("task.tokens_per_sample")


@register_model("transformer_lm", dataclass=TransformerLanguageModelConfig)
class TransformerLanguageModel(FairseqLanguageModel):
    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_fastbpe(path):
            return {"path": path, "tokenizer": "moses", "bpe": "fastbpe"}

        def spm(path):
            return {"path": path, "tokenizer": "space", "bpe": "sentencepiece"}

        return {
            "transformer_lm.gbw.adaptive_huge": "https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2",
            "transformer_lm.wiki103.adaptive": "https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2",
            "transformer_lm.wmt19.en": moses_fastbpe("https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2"),
            "transformer_lm.wmt19.de": moses_fastbpe("https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2"),
            "transformer_lm.wmt19.ru": moses_fastbpe("https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2"),
            "transformer_lm.wmt20.en": spm("https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.en.tar.gz"),
            "transformer_lm.wmt20.ta": spm("https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.ta.tar.gz"),
            "transformer_lm.wmt20.iu.news": spm("https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.iu.news.tar.gz"),
            "transformer_lm.wmt20.iu.nh": spm("https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.iu.nh.tar.gz"),
        }
        # fmt: on

    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(parser, TransformerLanguageModelConfig(), delete_default=True, with_prefix="")

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        if cfg.decoder.max_positions is None:
            cfg.decoder.max_positions = cfg.tokens_per_sample

        if cfg.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(cfg.character_filters),
                cfg.character_embedding_dim,
                cfg.decoder.embed_dim,
                cfg.char_embedder_highway_layers,
            )
        elif cfg.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                cfg.decoder.input_dim,
                cfg.adaptive_input_factor,
                cfg.decoder.embed_dim,
                options.eval_str_list(cfg.adaptive_input_cutoff, type=int),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(cfg, task.source_dictionary, cfg.decoder.input_dim)

        if cfg.tie_adaptive_weights:
            assert cfg.adaptive_input
            assert cfg.adaptive_input_factor == cfg.adaptive_softmax_factor
            assert cfg.adaptive_softmax_cutoff == cfg.adaptive_input_cutoff, "{} != {}".format(
                cfg.adaptive_softmax_cutoff, cfg.adaptive_input_cutoff
            )
            assert cfg.decoder.input_dim == cfg.decoder.output_dim

        decoder = TransformerDecoderBase(cfg, task.target_dictionary, embed_tokens, no_encoder_attn=True)
        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens


def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if safe_hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if safe_hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    # Model training is not stable without this
    args.decoder_normalize_before = True


@register_model_architecture("transformer_lm", "transformer_lm_big")
def transformer_lm_big(args):
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_wiki103")
@register_model_architecture("transformer_lm", "transformer_lm_baevski_wiki103")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = safe_getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 8)
    args.dropout = safe_getattr(args, "dropout", 0.3)
    args.adaptive_input = safe_getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = safe_getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = safe_getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = safe_getattr(args, "adaptive_softmax_cutoff", "20000,60000")
    args.adaptive_softmax_dropout = safe_getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = safe_getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = safe_getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)


@register_model_architecture("transformer_lm", "transformer_lm_gbw")
@register_model_architecture("transformer_lm", "transformer_lm_baevski_gbw")
def transformer_lm_baevski_gbw(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 512)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.no_decoder_final_norm = safe_getattr(args, "no_decoder_final_norm", True)
    transformer_lm_big(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt")
def transformer_lm_gpt(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_small")
def transformer_lm_gpt2_small(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_tiny")
def transformer_lm_gpt2_tiny(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 64)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 1)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_medium")
def transformer_lm_gpt2_medium(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1280)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 5120)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 36)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 20)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_big")
def transformer_lm_gpt2_big(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1600)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 6400)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 25)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_big_wide")
def transformer_lm_gpt2_big_wide(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_bigger")
def transformer_lm_gpt2_bigger(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_layers = getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


def base_gpt3_architecture(args):
    args.decoder_input_dim = args.decoder_embed_dim
    args.decoder_output_dim = args.decoder_embed_dim
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", args.decoder_embed_dim * 4)
    # GPT-3 used learned positional embeddings, rather than sinusoidal
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", True)
    args.dropout = safe_getattr(args, "dropout", 0.0)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.0)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.share_decoder_input_output_embed = True
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_small")
def transformer_lm_gpt3_small(args):
    # 125M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_medium")
def transformer_lm_gpt3_medium(args):
    # 350M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_large")
def transformer_lm_gpt3_large(args):
    # 760M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1536)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_xl")
def transformer_lm_gpt3_xl(args):
    # 1.3B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_2_7")
def transformer_lm_gpt3_2_7(args):
    # 2.7B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2560)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_6_7")
def transformer_lm_gpt3_6_7(args):
    # 6.7B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 4096)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_13")
def transformer_lm_gpt3_13(args):
    # 13B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 40)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 5120)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 40)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_175")
def transformer_lm_gpt3_175(args):
    # 175B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 96)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 12288)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 96)
    base_gpt3_architecture(args)
