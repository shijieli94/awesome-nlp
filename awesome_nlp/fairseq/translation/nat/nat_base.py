import logging
from functools import partial

from awesome_nlp import CONFIG_REGISTRY, register_models_and_tasks
from awesome_nlp.fairseq import augment_name
from awesome_nlp.fairseq.translation import FairseqTranslationTask

logger = logging.getLogger(__name__)


register_tasks = partial(register_models_and_tasks, models="nat_base", commands=["fairseq_train", "fairseq_generate"])

augment_distilled = partial(augment_name, suffix="-distilled")


@register_tasks(*augment_distilled("iwslt14_de_en", "iwslt14_en_de"))
class NATransformerIWSLT14(FairseqTranslationTask):
    def post_process_configs(self):
        # first post-process specific configs
        """Used for post-processing NAT specific arguments"""
        # we can use the relative path (to the `--save-dir`) in the argument
        # `--iter-decode-with-external-reranker`, here we replace it
        # also we can treat it as bool config, in this case, we use the
        # `reranker.pt` as the default checkpoint name of reranker
        if CONFIG_REGISTRY.safe_hasattr("--iter-decode-with-external-reranker"):
            if "--iter-decode-with-external-reranker" not in CONFIG_REGISTRY["_relative_paths_"]:
                CONFIG_REGISTRY["_relative_paths_"].append("--iter-decode-with-external-reranker")

        # go to parent
        super().post_process_configs()

    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # tasks
                "--task": "translation_lev",
                "--noise": "full_mask",
                "--eval-bleu-args::iter_decode_max_iter@int": "1",
                "--eval-bleu-args::iter_decode_with_beam@int": "1",
                "_debug_::--eval-bleu-args::iter_decode_length_format@str": "oracle",
                # model
                "--arch": "nonautoregressive_transformer_iwslt_de_en",
                "--dropout": "0.3",
                "--share-all-embeddings": True,
                "--encoder-learned-pos": True,
                "--decoder-learned-pos": True,
                "--pred-length-offset": True,
                "--length-loss-factor": "0.1",
                "--activation-fn": "gelu",
                "--apply-bert-init": True,
                # criterion
                "--criterion": "nat_loss",
                "--label-smoothing": "0.1",
                # optimizer
                "--optimizer": "adam",
                "_adam_::--adam-betas": str((0.9, 0.98)),
                "_adam_::--adam-eps": "1e-8",
                "_adam_::--weight-decay": "0.01",
                # lr_scheduler
                "--lr-scheduler": "inverse_sqrt",
                "_inverse_sqrt_::--warmup-updates": "10000",
                "_inverse_sqrt_::--warmup-init-lr": "1e-7",
                # dataset
                "_debug_::--max-tokens": ("8192", "1024"),
                "--validate-interval": "0",  # do not validate at end_of_epoch
                "_debug_::--validate-interval-updates": ("1000", "10"),
                # optimization
                "_debug_::--max-update": ("200000", "20"),
                "--clip-norm": "10.0",
                "--lr": "5e-4",
                "--stop-min-lr": "1e-9",
                # checkpoint
                "--no-epoch-checkpoints": True,  # do not save at end_of_epoch
                "--save-interval": "0",  # do not save at end_of_epoch
                "_debug_::--save-interval-updates": ("1000", "10"),
                "--keep-interval-updates": "5",
                "--patience": "20",
                # tokenizer
                "--tokenizer": "moses",
                "_moses_::--source-lang": self.source_lang,
                "_moses_::--target-lang": self.target_lang,
            }
        )
        return configs

    @property
    def generate(self):
        configs = super().generate
        configs.update(
            {
                # NAT configs
                "--task": "translation_lev",
                "--iter-decode-max-iter": "1",
                "--iter-decode-with-beam": "1",
                # tokenizer
                "--tokenizer": "moses",
                "_moses_::--source-lang": self.source_lang,
                "_moses_::--target-lang": self.target_lang,
            }
        )
        return configs


@register_tasks(*augment_distilled("wmt16_en_ro", "wmt16_ro_en"))
class NATransformerWMT16(NATransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "nonautoregressive_transformer_wmt_en_de",
                "--dropout": "0.3",
                # dataset, 32K batch size assuming only one GPU
                "_debug_::--max-tokens": ("16384", "1024"),
                # optimization
                "_debug_::--update-freq": ("2", "1"),
            }
        )
        return configs


@register_tasks(*augment_distilled("wmt14_de_en", "wmt14_en_de"))
class NATransformerWMT14(NATransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "nonautoregressive_transformer_wmt_en_de",
                "--dropout": "0.1",
                # dataset, 64K batch size assuming only one GPU
                "_debug_::--max-tokens": ("16384", "1024"),
                # optimization
                "--lr": "7e-4",
                "_debug_::--update-freq": ("4", "1"),
            }
        )
        return configs
