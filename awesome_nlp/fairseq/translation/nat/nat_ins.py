import logging
from functools import partial

from awesome_nlp import register_models_and_tasks
from awesome_nlp.fairseq import augment_name
from awesome_nlp.fairseq.translation.nat.nat_base import NATransformerIWSLT14

logger = logging.getLogger(__name__)

register_tasks = partial(register_models_and_tasks, models="nat_ins", commands=["fairseq_train", "fairseq_generate"])

augment_distilled = partial(augment_name, suffix="-distilled")


@register_tasks(*augment_distilled("iwslt14_de_en", "iwslt14_en_de"))
class InsertNATransformerIWSLT14(NATransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "insertion_transformer_iwslt_de_en",
                "--pred-length-type": "none",
                # task
                "--noise": "random_delete",
            }
        )
        return configs

    @property
    def generate(self):
        configs = super().generate
        configs.update({})
        return configs
