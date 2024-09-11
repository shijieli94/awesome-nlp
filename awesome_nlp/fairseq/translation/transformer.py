import logging
from functools import partial

from awesome_nlp import register_models_and_tasks
from awesome_nlp.fairseq import augment_suffix
from awesome_nlp.fairseq.translation import FairseqTranslationTask

logger = logging.getLogger(__name__)


register_tasks = partial(
    register_models_and_tasks, models="transformer", commands=["fairseq_train", "fairseq_generate"]
)

augment_distilled = partial(augment_suffix, suffix="-distilled", append=True)


@register_tasks(*augment_distilled("iwslt14_de_en", "iwslt14_en_de"))
class TransformerIWSLT14(FairseqTranslationTask):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "transformer_iwslt_de_en",
                "--dropout": "0.3",
                "--share-all-embeddings": True,
                # tasks
                "--eval-bleu-args::beam@int": "5",
                "--eval-bleu-args::lenpen@float": "1",
                # criterion
                "--criterion": "label_smoothed_cross_entropy",
                "--label-smoothing": "0.1",
                "--report-accuracy": True,
                # optimizer
                "--optimizer": "adam",
                "_adam_::--adam-betas": str((0.9, 0.98)),
                "_adam_::--adam-eps": "1e-8",
                "_adam_::--weight-decay": "0.01",
                # lr_scheduler
                "--lr-scheduler": "inverse_sqrt",
                "_inverse_sqrt_::--warmup-updates": "4000",
                "_inverse_sqrt_::--warmup-init-lr": "1e-7",
                # dataset, 8K batch size assuming only one GPU
                "_debug_::--max-tokens": ("8192", "1024"),
                "--validate-interval": "0",  # do not validate at end_of_epoch
                "_debug_::--validate-interval-updates": ("1000", "10"),
                # optimization
                "_debug_::--max-update": ("30000", "20"),
                "--clip-norm": "10.0",
                "--lr": "5e-4",
                "--stop-min-lr": "1e-9",
                # checkpoint
                "--no-epoch-checkpoints": True,  # do not save at end_of_epoch
                "--save-interval": "0",  # do not save at end_of_epoch
                "_debug_::--save-interval-updates": ("1000", "10"),
                "--keep-interval-updates": "5",
                # tokenizer
                "--tokenizer": "moses",
                "_moses_::source_lang": self.source_lang,
                "_moses_::target_lang": self.target_lang,
            }
        )
        return configs

    @property
    def generate(self):
        configs = super().generate
        configs.update(
            {
                # generation
                "--beam": "5",
                "--lenpen": "1",
                # tokenizer
                "--tokenizer": "moses",
                "--source-lang": self.source_lang,
                "--target-lang": self.target_lang,
            }
        )
        return configs


@register_tasks(*augment_distilled("wmt16_en_ro", "wmt16_ro_en"))
class TransformerWMT16(TransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "transformer_wmt_en_de",
                "--dropout": "0.3",
                # dataset, 32K batch size assuming only one GPU
                "_debug_::--max-tokens": ("16384", "1024"),
                # optimization
                "_debug_::--update-freq": ("2", "1"),
            }
        )
        return configs


@register_tasks(*augment_distilled("wmt14_de_en", "wmt14_en_de"))
class TransformerWMT14(TransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "transformer_wmt_en_de",
                "--dropout": "0.1",
                # dataset, 64K batch size assuming only one GPU
                "_debug_::--max-tokens": ("16384", "1024"),
                # optimization
                "--lr": "7e-4",
                "_debug_::--update-freq": ("4", "1"),
            }
        )
        return configs
