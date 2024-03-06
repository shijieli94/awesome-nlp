import logging
import os
import random
from typing import List, Optional

import evaluate
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

DATASETS = {}


def register_dataset():
    def wrapper(cls):
        DATASETS[cls.name] = cls
        return cls

    return wrapper


def get_dataset(name, **kwargs):
    return DATASETS[name](**kwargs)


class Fields:
    def __init__(self, funcs=None):
        self.functions = {}
        if funcs is not None:
            self.functions.update(funcs)

    def add(self, key):
        def adder(func):
            self.functions[key] = func
            return func

        return adder

    def __contains__(self, item):
        return item in self.functions

    def __getitem__(self, name: str):
        return self.functions[name]

    def merge(self, field):
        new_field = Fields()
        new_field.functions = self.functions.update(new_field.functions)
        return new_field


class BaseDataset:
    name: str
    question_field: List[str]
    answer_field: str
    hf_path: str
    hf_name: Optional[str]
    metric: str
    fields: Fields
    gen_prefix: str = ""  # a small prefix before generating answer

    def __init__(self, dataset_split=None, dataset_path=None, ds_size=None, cache_dir=None):
        if isinstance(self.metric, str):
            self.metric = evaluate.load(self.metric, cache_dir=cache_dir)
        else:
            self.metric = self.metric(cache_dir=cache_dir)

        if dataset_path is None or not os.path.exists(dataset_path):
            self.dataset = load_dataset(self.hf_path, self.hf_name, cache_dir=cache_dir)
        else:
            self.dataset = Dataset.from_pandas(pd.read_json(dataset_path))
            logger.info(f"Loaded dataset from {dataset_path}, size {len(self.dataset)}")

        if dataset_split is not None and isinstance(self.dataset, DatasetDict):
            self.dataset = self.dataset[dataset_split]

        if ds_size is not None:
            logger.info(f"Loading partial dataset of size {ds_size}")
            self.dataset = load_partial_dataset(self.dataset, size=ds_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_field(self, entry, field):
        return self.fields[field](entry)

    def get_corpus(self, field):
        return [self.get_field(entry, field) for entry in self.dataset]


def load_partial_dataset(dataset, size=-1):
    # (0, 1) will return ratio, >= 1 will return exact size
    if size <= 0 or size >= len(dataset):
        return dataset

    total_size = len(dataset)
    size = int(size * total_size) if size < 1 else size

    rand = random.Random(x=size)
    index_list = list(range(total_size))
    rand.shuffle(index_list)
    dataset = dataset.select(index_list[:size])
    return dataset


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        with open(os.path.join(os.path.dirname(__file__), file), "r") as f:
            exec(f.read())
