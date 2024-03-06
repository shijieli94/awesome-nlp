import logging
import os

from ..utils import save_json
from . import deduplicate, encode_field
from .base_dsr import BaseDatasetReader

logger = logging.getLogger(__name__)


class RetrieverDatasetReader(BaseDatasetReader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # we need do deduplication in the index dataset, if dataset is loaded from local path,
        # it is already deduplicated, otherwise do deduplication in the question field
        dataset_path = kwargs.get("dataset_path")
        field = kwargs.get("field")

        if dataset_path is None or not os.path.exists(dataset_path):
            full_size = len(self)
            question_dataset = (
                self.encoded_dataset if field == "q" else encode_field(self.tokenizer, self.dataset_wrapper, "q")
            )
            keep_ids = deduplicate(question_dataset)
            if keep_ids is not None:
                self.dataset_wrapper.dataset = self.dataset_wrapper.dataset.select(keep_ids)
                self.encoded_dataset = self.encoded_dataset.select(keep_ids)

            logger.info(f"Keeping {len(self)}/{full_size} instances after deduplication")

            if dataset_path is not None:
                save_json(dataset_path, list(self.dataset_wrapper))
                logger.info(f"index dataset has been saved to {dataset_path}")

    def encode_dataset(self, field):
        # retriever dataset will add special tokens
        self.encoded_dataset = encode_field(self.tokenizer, self.dataset_wrapper, field, add_special_tokens=True)
