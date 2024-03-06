import multiprocessing
import os
from functools import partial

import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ..utils import save_json


def set_bm25(bm25):
    global bm25_global
    bm25_global = bm25


def search(args, num_candidates=1, num_ice=1):
    idx, query = args

    scores = bm25_global.get_scores(query)
    near_ids = list(np.argsort(scores)[::-1][: max(num_candidates, num_ice)])
    near_ids = [int(a) for a in near_ids]

    # add topk as one of the candidates
    ctxs_candidates = [near_ids[:num_ice]]
    while len(ctxs_candidates) < num_candidates:
        # ordered by sim score
        samples_ids = np.random.choice(len(near_ids), num_ice, replace=False)
        samples_ids = sorted(samples_ids)
        candidate = [near_ids[i] for i in samples_ids]
        if candidate not in ctxs_candidates:
            ctxs_candidates.append(candidate)
    return idx, ctxs_candidates[0], ctxs_candidates


class BM25Retriever:
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

        self.index_corpus = [
            word_tokenize(" ".join(text)) for text in self.index_wrapper.get_corpus(index_reader.field)
        ]
        self.dataset_corpus = [
            word_tokenize(" ".join(text)) for text in self.dataset_wrapper.get_corpus(dataset_reader.field)
        ]
        self.bm25 = BM25Okapi(self.index_corpus)

    def retrieve(self):
        if not os.path.exists(self.output_file) or self.overwrite_cache:
            pool = multiprocessing.Pool(processes=min(16, os.cpu_count()), initializer=set_bm25, initargs=(self.bm25,))
            func = partial(search, num_candidates=self.num_candidates, num_ice=self.num_ice)

            entries = list(self.dataset_wrapper.dataset)

            cntx_post = []
            with tqdm(total=len(self.dataset_corpus)) as pbar:
                for i, res in enumerate(pool.imap_unordered(func, list(enumerate(self.dataset_corpus)))):
                    pbar.update()
                    cntx_post.append(res)

            for idx, ctxs, ctxs_candidates in cntx_post:
                entries[idx]["ctxs"] = ctxs
                entries[idx]["ctxs_candidates"] = ctxs_candidates

            save_json(self.output_file, entries)
