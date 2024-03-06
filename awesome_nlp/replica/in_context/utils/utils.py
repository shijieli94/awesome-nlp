import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)

OPENAI_MODELS = ["gpt3", "code-davinci", "text-davinci"]


def save_json(file, data_list, indent=None):
    logger.info(f"Saving to {file}")
    if indent:  # convert to string to make json pretty-printed
        data_list = [{k: str(v) for k, v in d.items()} for d in data_list]

    with open(file, "w") as f:
        json.dump(data_list, f, indent=indent)


def load_json(file):
    logger.info(f"Loading from {file}")
    with open(file) as f:
        data = json.load(f)
    return data


def show_statistics(encoded_dataset, dataset_name):
    all_lens = [item["metadata"]["len"] for item in encoded_dataset]
    all_lens = pd.Series(all_lens, dtype=int)
    logger.info(f"length of {dataset_name}: {str(all_lens.describe())}")


def get_tokenizer(model_name, cache_dir=None):
    for model in OPENAI_MODELS:
        if model in model_name:
            from transformers import GPT2Tokenizer

            return GPT2Tokenizer.from_pretrained("gpt2-large", cache_dir=cache_dir)

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
