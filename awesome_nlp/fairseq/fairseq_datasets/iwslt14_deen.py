import logging
import os
import shutil

from awesome_nlp import DATASETS_DIR, ConfigRegistry
from awesome_nlp.fairseq import Dataset, augment_suffix

logger = logging.getLogger(__name__)


@Dataset.register(*augment_suffix("iwslt14_de_en", "iwslt14_en_de", suffix="-distilled", append=True))
class IWSLT2014DEEN(Dataset):
    NAME = "fairseq_iwslt14_deen"
    HF_PATH = "shijli/fairseq_iwslt14_deen"
    TOKENIZER = "moses"
    BPE = "subword_nmt"
    DESCRIPTION = "preprocessed with the script `prepare-iwslt14.sh` from fairseq"

    @classmethod
    def _load(cls, task, src_lang=None, tgt_lang=None, joined=False, **kwargs):
        if "-distilled" not in task:
            task = "iwslt14_de_en"  # name for raw dataset

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

            if "-distilled" in task:
                # distilled dataset share the same vocabulary as the original dataset, so we reuse the dict.
                dict_dir = binarized_dir.replace("-distilled", "")
                dict_dir = dict_dir.replace("iwslt14_en_de", "iwslt14_de_en")  # raw dataset name

                preprocess_args.quiet_update(
                    {
                        "--srcdict": os.path.join(dict_dir, "dict.de.txt"),
                        "--tgtdict": os.path.join(dict_dir, "dict.en.txt"),
                    }
                )

            from fairseq_cli import preprocess

            preprocess.cli_main(preprocess_args.to_args_list())

            shutil.rmtree(data_dir)

        return binarized_dir
