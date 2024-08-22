import os
import typing as t

import datasets
import torch
from torch.utils.data.dataset import Dataset


class IWSLT(Dataset):
    def __init__(
        self,
        version: str = "17",
        src_lang: str = "en",
        tgt_lang: str = "ro",
        data_dir: str = None,
        huggingface_tokenizer=None,
        split: str = None,
    ):
        self.version = version
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = 511

        self.dl = datasets.DownloadManager()

        self.name = f"iwslt{self.version}"

        self.version2folder = {
            "15": os.path.join(data_dir, "2015-01/texts"),
            "17": os.path.join(data_dir, "2017-01-trnted/texts"),
        }
        self.version2years = {
            "15": {
                "train_and_test": [2010, 2011, 2012, 2013],
                "dev": [2010],
            },
            "17": {
                "train_and_test": [2010, 2011, 2012, 2013, 2014, 2015],
                "dev": [2010],
            },
        }

        data_file = f"{self.version2folder[version]}/{src_lang}/{tgt_lang}/{src_lang}-{tgt_lang}.tgz"

        split_generators = self._split_generators(data_file)
        self.translation_dataset = self.load_dataset(split_generators, split=split)

        with torch.no_grad():
            self.tokenizer = huggingface_tokenizer

    def load_dataset(self, split_generators: t.List[datasets.SplitGenerator], split: str) -> t.List[t.Dict]:
        split_generators = self.concat_dataset(split_generators, split)
        return list(
            self._generate_examples(
                source_files=split_generators.gen_kwargs["source_files"],
                target_files=split_generators.gen_kwargs["target_files"],
            )
        )

    @staticmethod
    def concat_dataset(split_generators: t.List[datasets.SplitGenerator], split: str) -> datasets.SplitGenerator:
        split2ix = {"train": 0, "test": 1, "validation": 2}
        assert split in split2ix, "Iwslt: split must be either train or test on validation"
        if split is not None:
            return split_generators[split2ix[split]]

    def _split_generators(self, data_file: str) -> t.List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        pair = f"{self.src_lang}-{self.tgt_lang}"
        dl_dir = self.dl.extract(data_file)
        data_dir = os.path.join(dl_dir, f"{self.src_lang}-{self.tgt_lang}")

        years = self.version2years[self.version]["train_and_test"]
        dev = self.version2years[self.version]["dev"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_files": [os.path.join(data_dir, f"train.tags.{pair}.{self.src_lang}")],
                    "target_files": [os.path.join(data_dir, f"train.tags.{pair}.{self.tgt_lang}")],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_files": [
                        os.path.join(data_dir, f"IWSLT{self.version}.TED.tst{year}.{pair}.{self.src_lang}.xml")
                        for year in years
                    ],
                    "target_files": [
                        os.path.join(data_dir, f"IWSLT{self.version}.TED.tst{year}.{pair}.{self.tgt_lang}.xml")
                        for year in years
                    ],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "source_files": [
                        os.path.join(data_dir, f"IWSLT{self.version}.TED.dev{year}.{pair}.{self.src_lang}.xml")
                        for year in dev
                    ],
                    "target_files": [
                        os.path.join(data_dir, f"IWSLT{self.version}.TED.dev{year}.{pair}.{self.tgt_lang}.xml")
                        for year in dev
                    ],
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, source_files: t.List[str], target_files: t.List[str]) -> t.List[t.Dict]:
        """Yields examples."""
        for source_file, target_file in zip(source_files, target_files):
            with open(source_file, "r", encoding="utf-8") as sf:
                with open(target_file, "r", encoding="utf-8") as tf:
                    for source_row, target_row in zip(sf, tf):
                        source_row = source_row.strip()
                        target_row = target_row.strip()

                        if source_row.startswith("<"):
                            if source_row.startswith("<seg"):
                                # Remove <seg id="1">.....</seg>
                                # Very simple code instead of regex or xml parsing
                                part1 = source_row.split(">")[1]
                                source_row = part1.split("<")[0]
                                part1 = target_row.split(">")[1]
                                target_row = part1.split("<")[0]

                                source_row = source_row.strip()
                                target_row = target_row.strip()
                            else:
                                continue

                        yield {
                            "translation": {
                                self.src_lang: source_row,
                                self.tgt_lang: target_row,
                            }
                        }

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
