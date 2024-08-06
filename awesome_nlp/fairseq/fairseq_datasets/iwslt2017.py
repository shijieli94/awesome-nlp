import logging
import os
import shutil

from awesome_nlp import DATASETS_DIR, TRANSFORMERS_DIR, ConfigRegistry
from awesome_nlp.fairseq import Dataset, augment_suffix

logger = logging.getLogger(__name__)


@Dataset.register("iwslt17_en_zh")
class TedIWSLT2014(Dataset):
    NAME = "fairseq_iwslt2017"
    HF_PATH = "iwslt2017"
    TOKENIZER = "bert"
    BPE = None
    DESCRIPTION = "tokenized with bert-base-multilingual-cased but lowercased"

    @classmethod
    def _load(cls, task, src_lang=None, tgt_lang=None, joined=False, **kwargs):
        lang1, lang2 = task.split("_")[-2:]
        if lang2 == "en":
            lang1, lang2 = lang2, lang1

        task_dir = os.path.join(DATASETS_DIR, cls.NAME, f"{lang1}-{lang2}")
        os.makedirs(task_dir, exist_ok=True)

        binarized_dir = os.path.join(task_dir, "binarized-joined" if joined else "binarized")

        if not os.path.exists(binarized_dir):
            data_dir = os.path.join(task_dir, "tokenized")

            # cls._tokenize(lang1, lang2, data_dir)

            preprocess_args = ConfigRegistry()

            preprocess_args.quiet_update(
                {
                    "--source-lang": lang1,
                    "--target-lang": lang2,
                    "--trainpref": os.path.join(data_dir, "train"),
                    "--validpref": os.path.join(data_dir, "valid"),
                    "--testpref": os.path.join(data_dir, "test"),
                    "--destdir": binarized_dir,
                    "--workers": str(os.cpu_count()),
                    "--joined-dictionary": joined,
                }
            )

            from fairseq_cli import preprocess

            preprocess.cli_main(preprocess_args.to_args_list())

            shutil.rmtree(data_dir)

        return binarized_dir

    @classmethod
    def _tokenize(cls, lang1, lang2, data_dir):
        import datasets
        from tqdm import tqdm
        from transformers import AutoTokenizer

        os.makedirs(data_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir=TRANSFORMERS_DIR)

        data = datasets.load_dataset(
            cls.HF_PATH, pair=f"{lang1}-{lang2}", is_multilingual=False, cache_dir=DATASETS_DIR
        )

        for split in ["train", "validation", "test"]:
            name = "valid" if split == "validation" else split
            for lang in [lang1, lang2]:
                with open(os.path.join(data_dir, f"{name}.{lang}"), "w") as f:
                    for text in tqdm(data[split]["translation"]):
                        f.write(" ".join(tokenizer.tokenize(text[lang].lower())) + "\n")
