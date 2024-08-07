import logging
from functools import partial

from awesome_nlp import CONFIG_REGISTRY, TRANSFORMERS_DIR, register_models_and_tasks
from awesome_nlp.fairseq import augment_suffix
from awesome_nlp.fairseq.translation.nat.nat_base import NATransformerIWSLT14

logger = logging.getLogger(__name__)

register_tasks = partial(
    register_models_and_tasks,
    models=["dacrf_transformer", "dacrf_transformer_finetune"],
    commands=["fairseq_train", "fairseq_generate"],
)

augment_distilled = partial(augment_suffix, suffix="-distilled", append=True)


@register_tasks(*augment_distilled("iwslt14_de_en", "iwslt14_en_de"))
class DACRFTransformerIWSLT14(NATransformerIWSLT14):
    def post_process_configs(self):
        if "finetune" in self.model and "train" in self.command:
            logger.warning("add finetuning configs...")
            CONFIG_REGISTRY.update(self.finetune)
            CONFIG_REGISTRY.update({"_polynomial_decay_::--total-num-update": CONFIG_REGISTRY["--max-update"]})

        if CONFIG_REGISTRY.safe_hasattr("--glance-p"):
            glance_p = CONFIG_REGISTRY.pop("--glance-p").replace("{max_update}", CONFIG_REGISTRY["--max-update"])
            CONFIG_REGISTRY.update({"--glance-p": glance_p})

        if CONFIG_REGISTRY.safe_gt("--dacrf-loss-factor", 0.0):
            CONFIG_REGISTRY.update(
                {
                    "--crf-lowrank-approx": "64",
                    "--crf-beam-approx": "64",
                    "--crf-decode-beam": "4",
                }
            )

        # if CONFIG_REGISTRY.safe_gt("--upsample-scale", 4.0):
        #     logger.warning("Reducing to small batch size due to large upsample scale")
        #     CONFIG_REGISTRY.update({"--max-tokens": str(int(CONFIG_REGISTRY["--max-tokens"]) // 2)})
        #     CONFIG_REGISTRY.update({"--update-freq": str(int(CONFIG_REGISTRY["--update-freq"]) * 2)})

        # go to parent
        super().post_process_configs()

    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # task
                "--filter-max-sizes": "256,1024",
                "--filter-ratios": "2",
                "--skip-invalid-size-inputs-valid-test": True,
                # model
                "--arch": "dacrf_transformer_iwslt_de_en",
                "--upsample-scale": "2",
                "--upsample-target-length": False,
                "--decode-strategy": "viterbi",
                "--glance-p": "0.5@0-0.1@{max_update}",
                # CRF configs
                "--dag-loss-factor": "1.0",
                "--dacrf-loss-factor": "0.0",
            }
        )
        return configs

    @property
    def finetune(self):
        configs = {
            # model
            "--finetune-from-model": "checkpoint_finetune.pt",
            "--no-strict-model-load": True,
            "--fix-dag-params": True,
            "--glance-p": None,
            "--length-loss-factor": "0.0",
            "--dag-loss-factor": "0.0",
            "--dacrf-loss-factor": "1.0",
            "--lr": "5e-4",
            "--lr-scheduler": "polynomial_decay",
            "_polynomial_decay_::--warmup-updates": "0",
            "_polynomial_decay_::--end-learning-rate": "1e-5",
            "_polynomial_decay_::--power": "1",
        }
        return configs


@register_tasks(*augment_distilled("wmt16_en_ro", "wmt16_ro_en"))
class DACRFTransformerWMT16(DACRFTransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "dacrf_transformer_wmt_en_de",
                "--dropout": "0.3",
                # dataset, 32K batch size assuming only one GPU
                "_debug_::--max-tokens": ("16384", "1024"),
                # optimization
                "_debug_::--update-freq": ("2", "1"),
            }
        )
        return configs


@register_tasks(*augment_distilled("wmt14_de_en", "wmt14_en_de", "wmt17_en_zh", "wmt17_zh_en"))
class NATransformerWMT14(DACRFTransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "dacrf_transformer_wmt_en_de",
                "--dropout": "0.1",
                # dataset, 64K batch size assuming only one GPU
                "_debug_::--max-tokens": ("16384", "1024"),
                # optimization
                "_debug_::--update-freq": ("4", "1"),
            }
        )
        if self.task in ["wmt17_en_zh", "wmt17_zh_en"]:
            configs.update(
                {
                    "--share-all-embeddings": False,
                    "--share-decoder-input-output-embed": True,
                }
            )
        return configs

    @property
    def finetune(self):
        configs = super().finetune
        configs.update(
            {
                "--max-tokens": "65536",
                "--update-freq": "1",
            }
        )
        return configs


document_mt = [
    "iwslt17_en_de",
    "nc2016_en_de",
    "europarl7_en_de",
    "iwslt17_de_en",
    "nc2016_de_en",
    "europarl7_de_en",
]


@register_tasks(*augment_suffix(document_mt, suffix="-sent"))
@register_tasks("iwslt17_en_zh", "iwslt17_zh_en")
class DACRFTransformerSent(DACRFTransformerIWSLT14):
    def __init__(self, command, model, task):
        super().__init__(command, model, task)
        self.use_bert = self.task in ["iwslt17_en_zh", "iwslt17_zh_en"]

    def post_process_configs(self):
        if "europarl7" in self.task and "train" in self.command:
            CONFIG_REGISTRY.update(self.europarl7_configs)

        # go to parent
        super().post_process_configs()

    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # task
                "--prepend-bos-src": True,  # already exists in source sentences
                "--prepend-bos-tgt": True,  # already exists in target sentence
                # model
                "--arch": "dacrf_transformer_wmt_en_de",
                "--num-workers": "4",
            }
        )
        if self.use_bert:
            configs.update(
                {
                    "--share-all-embeddings": False,
                    "--share-decoder-input-output-embed": True,
                    "--tokenizer": "bert",
                    "_bert_::bpe_model_name": "bert-base-multilingual-cased",
                    "_bert_::bpe_cache_dir": TRANSFORMERS_DIR,
                }
            )
        else:
            configs.update(
                {
                    "--filter-max-sizes": None,
                    "--filter-ratios": None,
                }
            )
        return configs

    @property
    def generate(self):
        configs = super().generate
        if self.use_bert:
            configs.update(
                {
                    "--tokenizer": "bert",
                    "--sacrebleu-tokenizer": "zh",
                    "--bpe-model-name": "bert-base-multilingual-cased",
                    "--bpe-cache-dir": TRANSFORMERS_DIR,
                }
            )
        return configs

    @property
    def europarl7_configs(self):
        configs = {
            "--update-freq": "2",
            "--dropout": "0.2",
        }
        return configs


@register_tasks(*augment_suffix(document_mt, suffix="-doc"))
class GroupDACRFTransformerDoc(DACRFTransformerIWSLT14):
    def post_process_configs(self):
        if "europarl7" in self.task and "train" in self.command:
            CONFIG_REGISTRY.update(self.europarl7_configs)

        # go to parent
        super().post_process_configs()

    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                "--filter-max-sizes": None,
                "--filter-ratios": None,
                # task
                "--prepend-bos-src": True,  # already exists in source sentences
                "--prepend-bos-tgt": True,  # already exists in target sentence
                # model
                "--arch": "dacrf_transformer_doc_wmt_en_de",
                "--num-workers": "4",
                "--encoder-ctxlayers": "2",
                "--decoder-ctxlayers": "0",  # use local for all decoder layer
                "--cross-ctxlayers": "0",  # use local for all cross layer
                "--eval-bleu-args::iter_decode_force_max_iter@bool": True,
                "--eval-bleu-args::iter_decode_max_iter@int": "1",
                # dataset
                "_debug_::--max-tokens": ("4096", "1024"),
                "_debug_::--update-freq": ("2", "1"),
            }
        )
        return configs

    @property
    def generate(self):
        configs = super().generate
        configs.update(
            {
                # model
                "--iter-decode-force-max-iter": True,
                "--iter-decode-max-iter": "1",
                "--batch-size": "4",
            }
        )
        return configs

    @property
    def europarl7_configs(self):
        configs = {
            "--max-tokens": "2048",
            "--update-freq": "8",
            "--dropout": "0.2",
        }
        return configs
