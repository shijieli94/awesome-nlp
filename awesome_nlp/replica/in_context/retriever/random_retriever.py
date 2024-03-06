import logging
import os

import numpy as np
from tqdm import tqdm

from ..utils import save_json

logger = logging.getLogger(__name__)


class RandomRetriever:
    def __init__(
        self,
        dataset_reader,
        index_reader,
        num_ice,
        num_candidates,
        output_file,
        overwrite_cache=False,
        **kwargs,
    ) -> None:
        self.dataset_wrapper = dataset_reader.dataset_wrapper
        self.index_wrapper = index_reader.dataset_wrapper

        self.num_ice = num_ice
        self.num_candidates = num_candidates
        self.output_file = output_file
        self.overwrite_cache = overwrite_cache

    def _retrieve(self):
        num_index = len(self.index_wrapper)
        ctxs_candidates = []
        while len(ctxs_candidates) < self.num_candidates:
            candidate = np.random.choice(num_index, self.num_ice, replace=False).tolist()
            if candidate not in ctxs_candidates:
                ctxs_candidates.append(candidate)
        return ctxs_candidates[0], ctxs_candidates

    def retrieve(self):
        if os.path.exists(self.output_file) and not self.overwrite_cache:
            return

        entries = []
        for entry in tqdm(self.dataset_wrapper):
            ctxs, ctxs_candidates = self._retrieve()
            entry["ctxs"] = ctxs
            entry["ctxs_candidates"] = ctxs_candidates
            entries.append(entry)

        save_json(self.output_file, entries)
