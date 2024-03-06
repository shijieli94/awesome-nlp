import logging
import os
from functools import partial

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import DataCollatorWithPaddingAndCuda, save_json

logger = logging.getLogger(__name__)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask, norm_embed=True):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sentence_embed = torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)
    if norm_embed:
        sentence_embed = torch.nn.functional.normalize(sentence_embed, p=2, dim=1)
    return sentence_embed


class DenseRetriever:
    def __init__(
        self,
        model,
        searcher,
        dataset_reader,
        index_reader,
        num_ice,
        num_candidates,
        output_file,
        overwrite_cache=False,
        batch_size=64,
        norm_embed=True,
        faiss_index=None,
        accelerator=None,
        **kwargs,
    ) -> None:
        self.dataset_reader = dataset_reader
        self.index_reader = index_reader

        self.num_ice = num_ice
        self.num_candidates = num_candidates
        self.output_file = output_file
        self.batch_size = batch_size
        self.norm_embed = norm_embed
        self.overwrite_cache = overwrite_cache

        self.searcher = partial(searcher, num_ice=num_ice, num_candidates=num_candidates)

        self.model = model.eval()
        self.model = accelerator.prepare(self.model)
        if hasattr(self.model, "module"):
            self.model = self.model.module

        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.model.device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=batch_size, collate_fn=co)
        if os.path.exists(faiss_index) and not self.overwrite_cache:
            logger.info(f"Loading faiss index from {faiss_index}")
            self.index = faiss.read_index(faiss_index)
        else:
            self.index = self.create_index(faiss_index)

    def create_index(self, faiss_index):
        logger.info("Building faiss index...")
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.index_reader.tokenizer, device=self.model.device)
        dataloader = DataLoader(self.index_reader, batch_size=self.batch_size, collate_fn=co)

        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        res_list = self.forward(dataloader)
        id_list = np.array([res["metadata"]["id"] for res in res_list])
        embed_list = np.stack([res["embed"] for res in res_list])
        index.add_with_ids(embed_list, id_list)
        faiss.write_index(index, faiss_index)
        logger.info(f"Saving faiss index to {faiss_index}, size {len(self.index_reader)}")
        return index

    def forward(self, dataloader):
        res_list = []
        for i, entry in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                model_output = self.model(**entry)
                embeddings = mean_pooling(model_output, entry["attention_mask"], norm_embed=self.norm_embed)
            res_list.extend([{"embed": embed.tolist(), "metadata": meta} for embed, meta in zip(embeddings, metadata)])
        return res_list

    def retrieve(self):
        if not os.path.exists(self.output_file) or self.overwrite_cache:
            res_list = self.forward(self.dataloader)
            for res in res_list:
                res["entry"] = self.dataset_reader.dataset_wrapper[res["metadata"]["id"]]

            entries = self.searcher(res_list, self.index)
            save_json(self.output_file, entries)
