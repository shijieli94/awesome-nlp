import logging
from functools import partial

from awesome_nlp import register_models_and_tasks
from awesome_nlp.fairseq import augment_suffix
from awesome_nlp.fairseq.translation.nat.nat_base import NATransformerIWSLT14

logger = logging.getLogger(__name__)

register_tasks = partial(register_models_and_tasks, models="nat_crf", commands=["fairseq_train", "fairseq_generate"])

augment_distilled = partial(augment_suffix, suffix="-distilled", append=True)


@register_tasks(*augment_distilled("iwslt14_de_en", "iwslt14_en_de"))
class CRFTransformerIWSLT14(NATransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "nacrf_transformer_iwslt_de_en",
                "--crf-lowrank-approx": "32",
                "--crf-beam-approx": "64",
                "--word-ins-loss-factor": "0.5",
            }
        )
        return configs
