import typing as t

import datasets
import torch
from torch.utils.data import Dataset

from .. import DATASETS_DIR


class WMT(Dataset):
    """
    Wmt machine translation dataset reader

    Input:
        - version -> the dataset version dataset, by default '16' (dataset-16)
        - src_lang -> the source language, by default 'ro' (Romanian)
        - tgt_lang -> the target language, by default 'en' (English)
        - tokenizer_model -> the tokenizer model
        - split -> if not None, allows to split the dataset in following set: ['train', 'test', 'validation']
        - concat -> if not None, make possible the concatenation of the specified set.
                    Note: It works only if split is None
                    It can be: ['train', 'test', 'validation']
    """

    def __init__(
        self,
        version: str = "16",
        src_lang: str = "ro",
        tgt_lang: str = "en",
        huggingface_tokenizer=None,
        split: str = None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer_model = huggingface_tokenizer
        self.max_length = 511

        if src_lang == "en":
            source2target = "{}-{}".format(self.tgt_lang, self.src_lang)
        else:
            source2target = "{}-{}".format(self.src_lang, self.tgt_lang)

        if version == "19" and "test" in split:
            split = "validation"

        version = f"wmt{version}"

        self.name = version

        try:
            self.translation_dataset = datasets.load_dataset(
                version, source2target, split=split, cache_dir=DATASETS_DIR
            )
        except:
            raise ValueError(
                f"{version} can read only the pairs cs-en, en-cs, de-en, en-de,"
                f" fi-en, en-fi, ro-en, en-ro, ru-en, en-ru, tr-en, en-tr"
            )

        with torch.no_grad():
            self.tokenizer = huggingface_tokenizer

    def collate_fn(self, batch):
        batch_source = [b[0] for b in batch]
        batch_target = [b[1] for b in batch]

        encoded_source = self.tokenizer(
            batch_source,
            padding=True,
            return_tensors="pt",
        )
        encoded_target = self.tokenizer(
            batch_target,
            padding=True,
            return_tensors="pt",
        )

        return {
            "source": {
                "input_ids": encoded_source["input_ids"],
                "attention_mask": encoded_source["attention_mask"],
                "sentences": batch_source,
            },
            "target": {
                "input_ids": encoded_target["input_ids"],
                "attention_mask": encoded_target["attention_mask"],
                "sentences": batch_target,
            },
        }

    def __len__(self) -> int:
        return len(self.translation_dataset)

    def __getitem__(self, idx: int) -> t.Tuple[str, str]:
        sample = self.translation_dataset[idx]
        source = sample["translation"][self.src_lang]
        target = sample["translation"][self.tgt_lang]

        return source, target
