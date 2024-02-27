"""BM25 Retriever"""

import ast
import logging
import math
from collections import Counter
from typing import List, Optional

import numpy as np
from accelerate import Accelerator
from nltk.tokenize import word_tokenize
from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from tqdm import trange

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 In-context Learning Retriever Class
        Class of BM25 Retriever.

    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        index_corpus (:obj:`List[str]`) : A corpus created from the input field data of :obj:`index_ds`.
        test_corpus (:obj:`List[str]`) : A corpus created from the input field data of :obj:`test_ds`.
        bm25 (:obj:`BM250kapi`): An instance of :obj:`BM250kapi` class, initialized using :obj:`index_ds`.
    """

    bm25 = None
    index_corpus = None
    test_corpus = None

    def __init__(
        self,
        dataset_reader: DatasetReader,
        ice_separator: Optional[str] = "\n",
        ice_eos_token: Optional[str] = "\n",
        prompt_eos_token: Optional[str] = "",
        ice_num: Optional[int] = 1,
        index_split: Optional[str] = "train",
        test_split: Optional[str] = "test",
        accelerator: Optional[Accelerator] = None,
        # BM25Retriever
        move_nearest_to_end: bool = False,
        reranker: Optional[str] = None,
        reranker_kwargs: Optional[str] = None,
    ) -> None:
        super().__init__(
            dataset_reader,
            ice_separator,
            ice_eos_token,
            prompt_eos_token,
            ice_num,
            index_split,
            test_split,
            accelerator,
        )
        from rank_bm25 import BM25Okapi

        self.index_corpus = [
            word_tokenize(data) for data in self.dataset_reader.generate_input_field_corpus(self.index_ds)
        ]
        self.bm25 = BM25Okapi(self.index_corpus)
        self.test_corpus = [
            word_tokenize(data) for data in self.dataset_reader.generate_input_field_corpus(self.test_ds)
        ]
        self.move_nearest_to_end = move_nearest_to_end
        self.reranker = reranker
        self.reranker_kwargs = ast.literal_eval(reranker_kwargs) if reranker_kwargs is not None else {}

    def retrieve(self) -> List[List]:
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            query = self.test_corpus[idx]
            scores = self.bm25.get_scores(query)
            near_ids = list(np.argsort(scores)[::-1][: self.ice_num])
            near_ids = [int(a) for a in near_ids]

            if self.reranker is not None and self.reranker == "recall":
                rerank_idx = rerank_with_recall(
                    source=query, prompts=[self.index_corpus[i] for i in near_ids], **self.reranker_kwargs
                )
                near_ids = [near_ids[i] for i in rerank_idx]

            if self.move_nearest_to_end:
                near_ids = list(reversed(near_ids))

            rtr_idx_list.append(near_ids)

        logger.info(f"Average in-context examples: {sum(len(a) for a in rtr_idx_list) / len(rtr_idx_list)}")
        return rtr_idx_list


def extract_all_word_ngrams(tokens: List[str], min_order: int, max_order: int):
    ngrams = []

    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))

    return Counter(ngrams), len(tokens)


def get_recall_score(src_ngrams, prompt_ngrams, max_order, effective_order=True):
    correct = [0 for _ in range(max_order)]
    total = correct.copy()
    for ngram, count in src_ngrams.items():
        n = len(ngram) - 1
        total[n] += count
        if ngram in prompt_ngrams:
            correct[n] += min(count, prompt_ngrams[ngram])

    scores = [0.0 for _ in range(max_order)]
    smooth_mteval = 1.0
    eff_order = max_order
    if not any(correct):
        return 0.0
    for n in range(1, len(scores) + 1):
        if total[n - 1] == 0:
            break
        if effective_order:
            eff_order = n
        if correct[n - 1] == 0:
            smooth_mteval *= 2
            scores[n - 1] = 100.0 / (smooth_mteval * total[n - 1])
        else:
            scores[n - 1] = 100.0 * correct[n - 1] / total[n - 1]

    log = lambda x: -9999999999 if x == 0 else math.log(x)
    score = math.exp(sum([log(p) for p in scores[:eff_order]]) / eff_order)

    return score


def rerank_with_recall(source, prompts, decays=0.1, min_threshold=1, max_ngram_order=4):
    src_ngrams, src_len = extract_all_word_ngrams(source, 1, max_ngram_order)

    prompts_ngrams = {}
    for i, prompt in enumerate(prompts):
        prompts_ngrams[i] = extract_all_word_ngrams(prompt, 1, max_ngram_order)[0]

    reranked_prompts = []
    while len(reranked_prompts) < len(prompts):
        scores = []
        for i in prompts_ngrams:
            scores.append(get_recall_score(src_ngrams, prompts_ngrams[i], max_ngram_order))

        top = np.argmax(scores)

        if scores[top] < min_threshold:
            if len(reranked_prompts) == 0:  # at least one prompt is selected
                reranked_prompts.append(top)
            break

        assert top not in reranked_prompts, "top is already selected, please check the code!"
        reranked_prompts.append(top)

        # downweight found ngrams and reset selected prompt
        for ngram in prompts_ngrams[top].keys():
            prompts_ngrams[top][ngram] = 0
            if ngram in src_ngrams:
                src_ngrams[ngram] *= decays

    return reranked_prompts
