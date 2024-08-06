import itertools
import logging
import os
import shutil

from awesome_nlp import DATASETS_DIR, ConfigRegistry
from awesome_nlp.fairseq import Dataset, augment_suffix

logger = logging.getLogger(__name__)

dataset_names = []
for lang, level in itertools.product(["_en_de", "_de_en"], ["-sent", "-doc"]):
    dataset_names.extend(augment_suffix("iwslt17", "nc2016", "europarl7", suffix=lang + level))


@Dataset.register(*dataset_names)
class DocumentMT(Dataset):
    NAME = "fairseq_document_deen"
    HF_PATH = "shijli/fairseq_document_deen"
    TOKENIZER = "moses"
    BPE = "subword_nmt"
    DESCRIPTION = "Dataset preprocessing follows the description on https://github.com/baoguangsheng/nat-on-doc"

    @classmethod
    def _load(cls, task, src_lang=None, tgt_lang=None, joined=False, **kwargs):
        task = task.replace("de_en", "en_de")
        task_dir = os.path.join(DATASETS_DIR, cls.NAME, task)
        os.makedirs(task_dir, exist_ok=True)

        binarized_dir = os.path.join(task_dir, "binarized-joined" if joined else "binarized")

        if not os.path.exists(binarized_dir):
            data_dir = cls._get_extracted_dir(task)

            preprocess_args = ConfigRegistry()

            preprocess_args.quiet_update(
                {
                    "--source-lang": "de",
                    "--target-lang": "en",
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
