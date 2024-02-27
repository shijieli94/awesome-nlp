import random
from collections import defaultdict
from typing import Optional

import numpy as np
from accelerate import Accelerator
from openicl import DatasetReader
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from sklearn.metrics.pairwise import cosine_similarity


class VotekRetriever(TopkRetriever):
    """Vote-k In-context Learning Retriever Class
        Class of Vote-k Retriever.

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
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:``model``.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
        votek_k (:obj:`int`, optional): ``k`` value of Voke-k Selective Annotation Algorithm. Defaults to ``3``.
    """

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
        sentence_transformers_model_name: Optional[str] = "all-mpnet-base-v2",
        cache_dir: Optional[str] = None,
        batch_size: Optional[int] = 1,
        votek_k: Optional[int] = 3,
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
        )
        self.votek_k = votek_k
        self._init_votek_index(overlap_threshold=1)

    def _init_votek_index(self, overlap_threshold=None):
        vote_stat = defaultdict(list)

        for i, cur_emb in enumerate(self.embed_list):
            cur_emb = cur_emb.reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(self.embed_list, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-self.votek_k - 1 :]
            assert sorted_indices[-1] == i
            for idx in sorted_indices[:-1]:
                vote_stat[idx].append(i)

        votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)

        self.selected_indices = []
        unselected_indices = []

        for j in range(len(votes)):
            candidate_set = set(votes[j][1])
            flag = True
            for pre in range(j):
                pre_set = set(votes[pre][1])
                if len(candidate_set.intersection(pre_set)) >= overlap_threshold * len(candidate_set):
                    flag = False
                    break

            if flag:
                self.selected_indices.append(int(votes[j][0]))
            else:
                unselected_indices += int(votes[j][0])

            if len(self.selected_indices) >= self.ice_num:
                break

        if len(self.selected_indices) < self.ice_num:
            self.selected_indices += random.sample(unselected_indices, self.ice_num - len(self.selected_indices))

    def retrieve(self):
        return [self.selected_indices[:] for _ in range(len(self.test_ds))]
