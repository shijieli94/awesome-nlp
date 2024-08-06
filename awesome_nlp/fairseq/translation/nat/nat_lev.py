import logging
from functools import partial

from awesome_nlp import register_models_and_tasks
from awesome_nlp.fairseq import augment_suffix
from awesome_nlp.fairseq.translation.nat.nat_base import NATransformerIWSLT14

logger = logging.getLogger(__name__)

register_tasks = partial(register_models_and_tasks, models="nat_lev", commands=["fairseq_train", "fairseq_generate"])

augment_distilled = partial(augment_suffix, suffix="-distilled", append=True)


@register_tasks(*augment_distilled("iwslt14_de_en", "iwslt14_en_de"))
class LevNATransformerIWSLT14(NATransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "levenshtein_transformer_iwslt_de_en",
                # task
                "--noise": "random_delete",
                "--eval-bleu-args::iter_decode_max_iter@int": "10",
            }
        )
        return configs

    @property
    def generate(self):
        configs = super().generate
        configs.update(
            {
                "--iter-decode-max-iter": "10",
            }
        )
        return configs
