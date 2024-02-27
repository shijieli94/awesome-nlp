import logging
import math
from typing import Optional

import numpy as np
from accelerate import Accelerator
from openicl import DatasetReader
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever

logger = logging.getLogger(__name__)


class DPPRetriever(TopkRetriever):
    """DPP In-context Learning Retriever Class
        Class of DPP Retriever.
        Two-stage DPP is used, where first stage is to get results of TopK to reduce candidate sets
        checkout https://arxiv.org/abs/2302.05698 for details.

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
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`.
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
        seed (:obj:`int`, optional): Seed for the random number generator. (:obj:`random_state` in :obj:`sample_exact_k_dpp` method)
        scale_factor (:obj:`float`, optional): A factor when gets the kernel.
    """

    model = None

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
        # TopkRetriever
        sentence_transformers_model_name: Optional[str] = "all-mpnet-base-v2",
        cache_dir: Optional[str] = None,
        batch_size: Optional[int] = 1,
        index_file: Optional[str] = None,
        move_nearest_to_end: bool = False,
        # DPPRetriever
        candidate_num: Optional[int] = 100,
        scale_factor: Optional[float] = 0.1,
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
            sentence_transformers_model_name,
            cache_dir,
            batch_size,
            index_file,
            move_nearest_to_end,
        )
        self.candidate_num = candidate_num
        self.scale_factor = scale_factor

    def get_kernel(self, embed, candidates):
        near_reps = np.array([[self.index.index.reconstruct(i) for i in candidate] for candidate in candidates])

        # normalize first
        embed = embed / np.linalg.norm(embed, keepdims=True, axis=-1)
        near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=-1)

        # to make kernel-matrix non-negative
        rel_scores = np.einsum("bd,bkd->bk", embed, near_reps)
        rel_scores = (rel_scores + 1) / 2

        # to prevent overflow error
        rel_scores -= rel_scores.max(keepdims=True, axis=-1)

        # to balance relevance and diversity
        rel_scores = np.exp(rel_scores / (2 * self.scale_factor))

        # to make kernel-matrix non-negative
        sim_matrix = np.matmul(near_reps, near_reps.transpose(0, 2, 1))
        sim_matrix = (sim_matrix + 1) / 2

        kernel_matrix = rel_scores[:, None, :] * sim_matrix * rel_scores[:, :, None]
        return near_reps, rel_scores, kernel_matrix

    def retrieve(self):
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in range(len(res_list))]
        logger.info("Retrieving data for test set...")

        embed_list = np.stack([res["embed"] for res in res_list])
        near_ids_list = self.index.search(embed_list, self.candidate_num)[1].tolist()

        # DPP stage
        near_reps, rel_scores, kernel_matrix = self.get_kernel(embed_list, near_ids_list)

        for idx, near_ids in enumerate(near_ids_list):
            # MAP inference
            samples_ids = fast_map_dpp(kernel_matrix[idx], self.ice_num)

            # recover the original idx
            rtr_sub_list = [near_ids[i] for i in samples_ids]

            if self.move_nearest_to_end:
                rtr_sub_list = list(reversed(rtr_sub_list))
            rtr_idx_list[idx] = rtr_sub_list

        return rtr_idx_list


def fast_map_dpp(kernel_matrix, max_length):
    """
    fast implementation of the greedy algorithm
    reference: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp_test.py
    paper: Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(int(selected_item))
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        selected_items.append(int(selected_item))
    return np.array(selected_items)
