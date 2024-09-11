import json
import logging
import os

import datasets

from awesome_nlp import (
    CONFIG_REGISTRY,
    DATASETS_DIR,
    DEBUG_MODE,
    EXPERIMENT_DIR,
    WANDB_DISABLED,
    BaseTask,
    Headline,
    register_command,
)

logger = logging.getLogger(__name__)

OPTIMIZER = ["_adam_", "_nag_", "_adagrad_"]
LR_SCHEDULER = ["_inverse_sqrt_", "_polynomial_decay_", "_fixed_"]


def augment_suffix(*names, suffix, append=False):
    return list(f"{name}{suffix}" for name in names) + (list(names) if append else [])


def augment_prefix(*names, prefix, append=False):
    return list(f"{prefix}{name}" for name in names) + (list(names) if append else [])


class Dataset:
    NAME: str  # unique name for saving
    HF_PATH: str  # Huggingface URL
    TOKENIZER: str  # tokenizer used during preprocessing
    BPE: str  # subword encoding used during preprocessing
    DESCRIPTION: str  # dataset description

    REGISTRY = {}

    @staticmethod
    def register(*dataset_names):
        def wrapper(cls):
            for dataset in dataset_names:
                if dataset in cls.REGISTRY:
                    raise ValueError(f"Dataset {dataset} has been registered.")
                cls.REGISTRY[dataset] = cls
            return cls

        return wrapper

    @classmethod
    def load_from_name(cls, task, **kwargs):
        import datasets

        datasets.utils.logging.set_verbosity_info()
        datasets.utils.logging.disable_propagation()  # to prevent double logging

        logger.info(f"Loading dataset {task}...")
        destination = cls.REGISTRY[task]._load(task, **kwargs)
        logger.info("Dataset Loaded!")
        return destination

    @classmethod
    def _load(cls, task, **kwargs):
        raise NotImplementedError(f"Please implement the `_load` method in the {cls.__name__} class.")

    @classmethod
    def _get_extracted_dir(cls, *args, **kwargs):
        dataset_builder = datasets.load_dataset_builder(
            cls.HF_PATH, *args, **kwargs, token=True, cache_dir=DATASETS_DIR
        )

        download_config = datasets.DownloadConfig(
            cache_dir=dataset_builder._cache_downloaded_dir,
            use_etag=False,
            token=dataset_builder.token,
            storage_options=dataset_builder.storage_options,
        )
        dl_manager = datasets.DownloadManager(
            dataset_name=dataset_builder.dataset_name,
            download_config=download_config,
            data_dir=dataset_builder.config.data_dir,
            base_path=dataset_builder.base_path,
            record_checksums=False,
        )
        data_dir = dl_manager.download_and_extract(dataset_builder.config.data_url)

        return data_dir


class FairseqTask(BaseTask):
    """Base Config for all Fairseq Tasks"""

    @property
    def train(self):
        """some general configs"""
        configs = {
            # common
            "_debug_::--log-interval": ("100", "10"),  # the second value will be used in DEBUG mode
            "--log-format": "simple",
            "--log-file": "log.txt",  # note: add to _relative_paths_
            "--tensorboard-logdir": "tensorboard",  # note: add to _relative_paths_
            "--wandb-project": None if WANDB_DISABLED else f"fairseq-{self.task}",
            "--seed": "42",
            "--fp16": True,
            # distributed
            "--ddp-backend": "c10d",
            # dataset
            "--required-batch-size-multiple": "1",  # keep this, otherwise the last batch will be dropped
            "--train-subset": "train",
            "--valid-subset": "valid",
            "--gen-subset": "test",
            "--fixed-validation-seed": "7",
            # checkpoint
            "--save-dir": self.save_dir,  # root path of _relative_paths_ for fairseq_train
            "--keep-best-checkpoints": "5",
        }
        return configs

    @property
    def save_dir(self):
        return os.path.join(EXPERIMENT_DIR, "fairseq", self.task, self.model + ("-debug" if DEBUG_MODE else ""))

    def run_task(self, main):
        """all paths in _relative_paths will be related to --save-dir"""
        CONFIG_REGISTRY["_relative_paths_"] = [
            "--log-file",
            "--tensorboard-logdir",
            "--restore-file",
            "--continue-once",
            "--finetune-from-model",
            "--path",
        ]

        # before processing configs, we need to settle down all mode dependent configs
        mode_dependent_key = list(k for k in CONFIG_REGISTRY.keys() if k.startswith("_debug_"))

        if DEBUG_MODE and len(mode_dependent_key) > 0:
            logger.warning("Running in DEBUG mode. Some configs may differ from those in standard mode.")

        for key in mode_dependent_key:
            val = CONFIG_REGISTRY.pop(key)
            if not isinstance(val, tuple) and not DEBUG_MODE:
                continue

            if isinstance(val, tuple):
                # in this case, the first element is used in normal mode
                # and the second element is used in debug mode
                val = val[1] if DEBUG_MODE else val[0]

            key = key.replace("_debug_::", "")
            CONFIG_REGISTRY.update({key: val}, _verbose=DEBUG_MODE, _level="warning")

        super().run_task(main)

    def post_process_configs(self):
        if CONFIG_REGISTRY.safe_hasattr("--optimizer"):
            optim = CONFIG_REGISTRY["--optimizer"]

            optimizer_configs = CONFIG_REGISTRY.parse_group_configs(included=OPTIMIZER)
            optimizer_configs = optimizer_configs.pop(f"_{optim}_", {})
            with Headline(f"optimizer: {optim}", level=2):
                CONFIG_REGISTRY.update(optimizer_configs)

        if CONFIG_REGISTRY.safe_hasattr("--lr-scheduler"):
            ls = CONFIG_REGISTRY["--lr-scheduler"]

            ls_configs = CONFIG_REGISTRY.parse_group_configs(included=LR_SCHEDULER)
            ls_configs = ls_configs.pop(f"_{ls}_", {})
            with Headline(f"lr-scheduler: {ls}", level=2):
                CONFIG_REGISTRY.update(ls_configs)

        # take care of grouped configs
        # eval_bleu_args is required by this task
        required_fields = CONFIG_REGISTRY.pop("_required_fields_", None)
        grouped_configs = CONFIG_REGISTRY.parse_group_configs(required_fields)

        model_overrides = {}
        if "--model-overrides" in grouped_configs:
            model_overrides["model"] = grouped_configs.pop("--model-overrides")
        if "--task-overrides" in grouped_configs:
            model_overrides["task"] = grouped_configs.pop("--task-overrides")
        if len(model_overrides) > 0:
            CONFIG_REGISTRY.update({"--model-overrides": json.dumps(model_overrides)})

        for key, val in grouped_configs.items():
            CONFIG_REGISTRY.update({key: json.dumps(val)})

        # take care of the relative path and suffix
        save_dir_suffix = CONFIG_REGISTRY.pop("--save-dir-suffix", "")

        if "train" in self.command and not DEBUG_MODE:
            os.environ["WANDB_NAME"] = self.model + save_dir_suffix

        # take care of relative path
        root_path = "--save-dir" if "train" in self.command else "--results-path"
        if save_dir_suffix:
            CONFIG_REGISTRY.update({root_path: CONFIG_REGISTRY[root_path] + save_dir_suffix})

        os.makedirs(CONFIG_REGISTRY[root_path], exist_ok=True)

        _abspath = lambda p: os.path.join(CONFIG_REGISTRY[root_path], p)
        for path in CONFIG_REGISTRY.pop("_relative_paths_", []):
            if CONFIG_REGISTRY.safe_hasattr(path):
                # multiple path will be separated by ":"
                path_list = CONFIG_REGISTRY[path].strip().split(",")
                path_list = [_abspath(p) for p in path_list]
                CONFIG_REGISTRY.update({path: os.pathsep.join(path_list)})


@register_command("fairseq_train", config="train")
def fairseq_train(input_args) -> None:
    from fairseq_cli.train import cli_main

    cli_main(input_args)


@register_command("fairseq_generate", config="generate")
def fairseq_generate(input_args) -> None:
    from fairseq_cli.generate import cli_main

    cli_main(input_args)


@register_command("fairseq_eval_lm", config="eval_lm")
def fairseq_eval_lm(input_args) -> None:
    from fairseq_cli.eval_lm import cli_main

    cli_main(input_args)
