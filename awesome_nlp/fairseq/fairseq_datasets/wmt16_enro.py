import logging
import os
import shutil

from awesome_nlp import DATASET_DIR, ConfigRegistry
from awesome_nlp.fairseq import Dataset, augment_name

logger = logging.getLogger(__name__)


@Dataset.register(*augment_name("wmt16_en_ro", "wmt16_ro_en", suffix="-distilled"))
class WMT2016ENRO(Dataset):
    PATH = os.path.join(DATASET_DIR, "fairseq", "wmt16_enro")
    HF_PATH = "shijli/wmt16_enro"
    TOKENIZER = "moses"
    BPE = "subword_nmt"
    DESCRIPTION = None

    @classmethod
    def _load(cls, task, src_lang=None, tgt_lang=None, joined=False, **kwargs):
        if "-distilled" not in task:
            task = "wmt16_en_ro"  # en-ro and ro-en use the same dataset

        task_dir = os.path.join(cls.PATH, task)
        os.makedirs(task_dir, exist_ok=True)

        binarized_dir = os.path.join(task_dir, "binarized-joined" if joined else "binarized")

        if not os.path.exists(binarized_dir):
            data_dir = cls._get_extracted_dir(task)

            preprocess_args = ConfigRegistry()

            preprocess_args.quiet_update(
                {
                    "--source-lang": "en",
                    "--target-lang": "ro",
                    "--trainpref": os.path.join(data_dir, "train"),
                    "--validpref": os.path.join(data_dir, "valid"),
                    "--testpref": os.path.join(data_dir, "test"),
                    "--destdir": binarized_dir,
                    "--workers": str(os.cpu_count()),
                    "--joined-dictionary": joined,
                }
            )

            if "-distilled" in task:
                # distilled dataset share the same vocabulary as the original dataset, so we reuse the dict.
                dict_dir = binarized_dir.replace("-distilled", "")
                dict_dir = dict_dir.replace("wmt16_ro_en", "wmt16_en_ro")  # raw dataset name

                preprocess_args.quiet_update(
                    {
                        "--srcdict": os.path.join(dict_dir, "dict.en.txt"),
                        "--tgtdict": os.path.join(dict_dir, "dict.ro.txt"),
                    }
                )

            from fairseq_cli import preprocess

            preprocess.cli_main(preprocess_args.to_args_list())

            shutil.rmtree(data_dir)

        return binarized_dir
