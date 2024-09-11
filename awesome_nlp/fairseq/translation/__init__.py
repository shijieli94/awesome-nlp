import json
import logging

import torch

from awesome_nlp import CONFIG_REGISTRY, Headline
from awesome_nlp.fairseq import Dataset, FairseqTask

logger = logging.getLogger(__name__)

TOKENIZER = ["_moses_", "_bert_"]
BPE = ["_subword_nmt_"]


class FairseqTranslationTask(FairseqTask):
    """Base Config for all Fairseq Translation Tasks"""

    def __init__(self, command, model, task):
        super().__init__(command, model, task)
        # register dataset
        self.dataset_cls = Dataset.REGISTRY[task]

        # task_name is in the form of `dataset_src_tgt-subtask`,
        # e.g. iwslt14_de_en or iwslt14_de_en-distilled.
        task, *_ = task.split("-")

        _, self.source_lang, self.target_lang = task.split("_")

    @property
    def train(self):
        """Basic translation configs and some general configs"""
        configs = super().train
        configs.update(
            {
                # tasks
                "--task": "translation",
                "--source-lang": self.source_lang,
                "--target-lang": self.target_lang,
                "--eval-bleu": True,
                "--eval-bleu-print-samples": True,
                # checkpoint
                "--best-checkpoint-metric": "bleu",
                "--maximize-best-checkpoint-metric": True,
                # used for post-processing
                "_required_fields_": ["--eval-bleu-args"],
            }
        )
        return configs

    @property
    def generate(self):
        configs = {
            # tasks
            "--task": "translation",
            "--source-lang": self.source_lang,
            "--target-lang": self.target_lang,
            # dataset
            "--batch-size": "128",
            "--required-batch-size-multiple": "1",
            # common_eval
            "--path": "checkpoint_best.pt",
            "--results-path": self.save_dir,  # root path of _relative_paths_ for fairseq_generate
            # scoring
            "--scoring": "sacrebleu",
            # used for post-processing
            "_required_fields_": [],
        }
        return configs

    def post_process_configs(self):
        if "train" in self.command:
            # adjust update frequency and max tokens based on visible device count
            device_count = torch.cuda.device_count()

            logger.warning(
                f"Found {device_count} cuda device(s), dynamically deciding batch size and update frequency..."
            )
            update_freq = int(CONFIG_REGISTRY.get("--update-freq", 1))
            max_tokens = int(CONFIG_REGISTRY["--max-tokens"])
            if update_freq > 1:
                max_tokens = str(max_tokens)
                update_freq = str(update_freq // device_count)
            else:
                max_tokens = str(max_tokens // device_count)
                update_freq = str(update_freq)

            CONFIG_REGISTRY.update({"--max-tokens": max_tokens, "--update-freq": update_freq}, _level="warning")

        # the first _positional_args_ should be data for fairseq translation task
        data_dir = self.dataset_cls.load_from_name(
            task=self.task,
            src_lang=self.source_lang,
            tgt_lang=self.target_lang,
            # as long as train configs have "--share-all-embeddings", we use joined dataset,
            # this is useful for determining dataset during inference
            joined=self.train.get("--share-all-embeddings", False),  # take care of inference
        )
        CONFIG_REGISTRY["_positional_args_"] = [data_dir]

        # we get tokenizer and bpe based on dataset_cls setting, but left an entries for changing it
        tokenizer = CONFIG_REGISTRY.pop("--tokenizer", self.dataset_cls.TOKENIZER)
        if tokenizer != self.dataset_cls.TOKENIZER:
            logger.warning(f"WARNING: tokenizer has been modified: {self.dataset_cls.TOKENIZER} -> {tokenizer}")

        tokenizer_configs = CONFIG_REGISTRY.parse_group_configs(included=TOKENIZER)
        tokenizer_configs = tokenizer_configs.pop(f"_{tokenizer}_", {})
        if tokenizer is not None:
            with Headline(f"tokenizer: {tokenizer}", level=2):
                if "train" in self.command:
                    # train stage use --eval-bleu-detok and --eval-bleu-detok-args
                    CONFIG_REGISTRY.update(
                        {"--eval-bleu-detok": tokenizer, "--eval-bleu-detok-args": json.dumps(tokenizer_configs)}
                    )
                else:
                    CONFIG_REGISTRY.update({"--tokenizer": tokenizer, **tokenizer_configs})

        bpe = CONFIG_REGISTRY.pop("--bpe", self.dataset_cls.BPE)
        if bpe != self.dataset_cls.BPE:
            logger.warning(f"WARNING: bpe has been modified: {self.dataset_cls.BPE} -> {bpe}")

        bpe_configs = CONFIG_REGISTRY.parse_group_configs(included=BPE)
        bpe_configs = bpe_configs.pop(f"_{bpe}_", {})
        if bpe is not None:
            with Headline(f"bpe: {bpe}", level=2):
                if "train" in self.command:
                    CONFIG_REGISTRY.update({"--eval-bleu-remove-bpe": bpe})
                else:
                    CONFIG_REGISTRY.update({"--post-process": bpe})

        # go to parent
        super().post_process_configs()
