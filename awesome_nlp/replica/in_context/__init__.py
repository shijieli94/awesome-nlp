import os
from pathlib import Path

from awesome_nlp import (
    CONFIG_REGISTRY,
    CONSTANTS,
    BaseTask,
    Headline,
    command_decorator,
    register_command,
)

from .dataset_readers.datasets import DATASETS

ROOT = os.path.join(CONSTANTS.EXPERIMENT_DIR, "in_context")

OPENAI_MODELS = ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"]
HF_MODELS = [
    "EleutherAI/gpt-neo-2.7B",
    # OPT models
    "facebook/opt-125m",
]
SENTENCE_TRANSFORMERS = ["google-bert/bert-base-uncased", "sentence-transformers/all-mpnet-base-v2"]

InContext_decorator = command_decorator("in_context")


class InContextBaseTask(BaseTask):
    @property
    def config(self):
        config = {
            "--seed": "42",
            "--model": "EleutherAI/gpt-neo-2.7B",
            # "--model": "text-davinci-003",
            "--sent_model": "bert-base-uncased",
            "--retrieve_ice": "50",
            "--inference_ice": "null",
            "--ntokens": "1600",
            "--half": True,
        }
        return config

    def _post_process_configs(self, run_dir):
        seed = CONFIG_REGISTRY["--seed"]
        model = CONFIG_REGISTRY["--model"]
        model_name = model.split("/")[-1]

        inference_ice = CONFIG_REGISTRY["--inference_ice"]
        retrieve_ice = CONFIG_REGISTRY["--retrieve_ice"]
        sentence_transformer = CONFIG_REGISTRY["--sent_model"]

        input_args = [
            f"hydra.run.dir={run_dir}",
            f"task_name={self.task}",
            f"model_name={model}",  # use full name
            f"cache_dir={CONSTANTS.TRANSFORMERS_CACHE_DIR}",
            f"index_file={os.path.join(run_dir, 'index_dataset.json')}",
            f"retrieved_file={os.path.join(run_dir, f'retrieved-ice{retrieve_ice}.json')}",
            f"predict_file={os.path.join(run_dir, f'preds-{model_name}-ice-{inference_ice}-{retrieve_ice}.json')}",
            f"batch_size={16 if CONFIG_REGISTRY['--half'] else 4}",
            f"seed={seed}",
            f"retrieve_ice={retrieve_ice}",
            f"inference_ice={inference_ice}",
            f"sentence_transformer={sentence_transformer}",
        ]
        if model in HF_MODELS:
            input_args.extend(
                [
                    f"inferencer_dsr.ntokens={CONFIG_REGISTRY['--ntokens']}",
                    f"inferencer_dsr.move_nearest_to_the_end=true",
                    f"inferencer.visualize_config.sample_ids={CONFIG_REGISTRY.get('--sample_ids', 'null')}",
                    f"inferencer.model_config.torch_dtype={'float16' if CONFIG_REGISTRY['--half'] else 'float32'}",
                ]
            )
        else:
            input_args.extend(
                [
                    "inferencer=api",
                    f"inferencer.model_config.keys_file={Path(__file__).parent.joinpath('openai_keys.txt').as_posix()}",
                    f"inferencer_dsr.ntokens=7000",
                    f"inferencer_dsr.ds_size=1000",
                ]
            )
        return input_args

    def run_task(self, main):
        Headline.log("POST-PROCESSING CONFIGS", level=3)
        input_args = self.post_process_configs()
        return main(input_args)


@InContext_decorator(*DATASETS.keys(), model="random")
class InContextRandomRetriever(InContextBaseTask):
    def post_process_configs(self):
        # random-seed42/mrpc/
        run_dir = os.path.join(ROOT, f"{self.model}-seed{CONFIG_REGISTRY['--seed']}", self.task)
        input_args = ["retriever=random"]
        input_args.extend(self._post_process_configs(run_dir=run_dir))
        return input_args


@InContext_decorator(*DATASETS.keys(), model="bm25")
class InContextBM25Retriever(InContextBaseTask):
    def post_process_configs(self):
        # random-seed42/mrpc/
        run_dir = os.path.join(ROOT, f"{self.model}-seed{CONFIG_REGISTRY['--seed']}", self.task)
        input_args = ["retriever=bm25"]
        input_args.extend(self._post_process_configs(run_dir=run_dir))
        return input_args


@InContext_decorator(*DATASETS.keys(), model=["dense", "dense_dpp"])
class InContextDenseRetriever(InContextBaseTask):
    def post_process_configs(self):
        # random-bert-base-uncased-seed42/mrpc/
        sent_model = CONFIG_REGISTRY["--sent_model"].split("/")[-1]
        run_dir = os.path.join(ROOT, f"{self.model}-{sent_model}-seed{CONFIG_REGISTRY['--seed']}", self.task)

        input_args = ["retriever=dense", f"retriever.faiss_index={os.path.join(run_dir, 'index.faiss')}"]
        if len(self.model.split("_")) == 2:
            input_args.append(f"retriever.method={self.model.split('_')[-1]}")

        input_args.extend(self._post_process_configs(run_dir=run_dir))
        return input_args


@register_command("in_context")
def main(input_args):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import datasets

    datasets.utils.logging.set_verbosity_info()
    datasets.utils.logging.disable_propagation()

    from .run import main as run_main

    run_main(input_args)
