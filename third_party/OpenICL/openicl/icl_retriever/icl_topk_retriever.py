import logging
import os
from typing import Optional

import faiss
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.icl_common_utils import get_dataloader
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TopkRetriever(BaseRetriever):
    """Topk In-context Learning Retriever Class
        Class of Topk Retriever.

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.move_nearest_to_end = move_nearest_to_end

        gen_datalist = self.dataset_reader.generate_input_field_corpus(self.test_ds)
        self.dataloader = get_dataloader(gen_datalist, batch_size)

        self.model = SentenceTransformer(sentence_transformers_model_name, cache_folder=cache_dir)
        self.model = self.model.to(self.device)
        self.model.eval()

        if index_file is not None and os.path.isfile(index_file):
            self.index = faiss.read_index(index_file)
        else:
            self.index = self.create_index(index_file)

    def create_index(self, index_file):
        self.select_datalist = self.dataset_reader.generate_input_field_corpus(self.index_ds)
        dataloader = get_dataloader(self.select_datalist, self.batch_size)

        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension()))
        res_list = self.forward(dataloader, process_bar=True, information="Creating index for index set...")
        id_list = np.array([res["id"] for res in res_list])
        self.embed_list = np.stack([res["embed"] for res in res_list])
        index.add_with_ids(self.embed_list, id_list)
        faiss.write_index(index, index_file)
        return index

    def forward(self, dataloader, process_bar=False, information=""):
        res_list = []
        if process_bar:
            logger.info(information)
            dataloader = tqdm.tqdm(dataloader, disable=not self.is_main_process)

        idx = 0
        for _, entry in enumerate(dataloader):
            with torch.no_grad():
                embeddings = self.model.encode(entry, show_progress_bar=False)

            for embed in embeddings:
                res_list.append({"embed": embed, "id": idx})
                idx += 1

        return res_list

    def retrieve(self):
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in res_list]
        logger.info("Retrieving data for test set...")

        embed_list = np.stack([res["embed"] for res in res_list])
        near_ids_list = self.index.search(embed_list, self.ice_num)[1].tolist()
        for idx, near_ids in enumerate(near_ids_list):
            if self.move_nearest_to_end:
                near_ids = list(reversed(near_ids))
            rtr_idx_list[idx] = near_ids
        return rtr_idx_list
