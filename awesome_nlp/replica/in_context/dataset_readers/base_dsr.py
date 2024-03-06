import logging

from torch.utils.data import Dataset

from ..utils import get_tokenizer
from . import encode_field
from .datasets import get_dataset

logger = logging.getLogger(__name__)


class BaseDatasetReader(Dataset):
    def __init__(
        self, task_name, model_name, field, dataset_split=None, dataset_path=None, ds_size=None, cache_dir=None
    ) -> None:
        self.field = field
        self.tokenizer = get_tokenizer(model_name, cache_dir=cache_dir)
        self.dataset_wrapper = get_dataset(
            task_name,
            dataset_split=dataset_split,
            dataset_path=dataset_path,
            ds_size=ds_size,  # index dataset use the full training set
            cache_dir=cache_dir,
        )
        self.encode_dataset(field)

    def encode_dataset(self, field):
        # base dataset do not add special tokens in order to get the actual length
        self.encoded_dataset = encode_field(self.tokenizer, self.dataset_wrapper, field, add_special_tokens=False)

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)
