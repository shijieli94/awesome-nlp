import glob
import json
import logging
import os
from pathlib import Path

import hydra.utils as hu
import torch
import tqdm
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ..utils import DataCollatorWithPaddingAndCuda, save_json, show_statistics

logger = logging.getLogger(__name__)


class HFInferencer:
    def __init__(
        self,
        retriever,
        model_config,
        dataset_reader,
        output_file,
        batch_size,
        visualize_config=None,
        **kwargs,
    ) -> None:
        self.accelerator = Accelerator()

        self.retriever = hu.instantiate(retriever, accelerator=self.accelerator)
        self.retriever.retrieve()

        self.dataset_reader = hu.instantiate(dataset_reader)
        self.dataset_reader.shard(self.accelerator)

        self.model = self._init_model(model_config)

        self.evaluator = self.dataset_reader.dataset_wrapper.metric

        if self.accelerator.is_main_process:
            logger.info(f"Statistics after sharding: ")
            show_statistics(self.dataset_reader.index_reader.encoded_dataset, "index dataset")
            show_statistics(self.dataset_reader.encoded_dataset, "main dataset")

        if visualize_config is not None:
            self.visualizer = hu.instantiate(
                visualize_config, output_dir=Path(output_file).with_suffix("").as_posix() + "-attentions"
            )
            if self.visualizer.enabled:
                logger.warning("Visualizing attention maps, batch size will be set to 1")
                batch_size = 1

        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.accelerator.device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=batch_size, collate_fn=co)

        self.label_to_ids = {}
        for key, label in self.dataset_reader.dataset_wrapper.fields["labels"]().items():
            # make sure a single mask is used, some tokenizers also consider space,
            # so we also consider here
            tokenized_label = self.dataset_reader.tokenizer.tokenize(" " + label)
            assert len(tokenized_label) == 1
            self.label_to_ids[key] = self.dataset_reader.tokenizer.convert_tokens_to_ids(tokenized_label[0])
        self.label_lists = list(self.label_to_ids)

        # OmegaConf DictConfig to dict
        self.generation_kwargs = OmegaConf.to_object(model_config.generation_kwargs)
        self.output_file = output_file

    def _init_model(self, model_config):
        model = hu.instantiate(model_config.model, torch_dtype=getattr(torch, model_config.torch_dtype))
        model.eval()
        self.model = self.accelerator.prepare(model)
        model.config.pad_token_id = self.dataset_reader.tokenizer.pad_token_id
        return model

    def forward(self):
        dataloader = self.dataloader
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(dataloader)

        avg_ice_num = 0
        res = []
        for i, entry in enumerate(dataloader):
            metadata = entry.pop("metadata")
            if "labels" in self.dataset_reader.dataset_wrapper.fields:
                # for classification tasks, we compare the ppl of provided generation_choices as generation
                with torch.no_grad():
                    outputs = self.model(**entry, output_attentions=True if self.visualizer.enabled else False)

                label_logits = outputs.logits.contiguous()[..., list(self.label_to_ids.values())]
                for mdata, logit in zip(metadata, label_logits):
                    mdata.pop("ctxs")
                    mdata.pop("ctxs_candidates")
                    label_positions = mdata.pop("label_positions")
                    label_logit = logit[label_positions, :]
                    preds = label_logit.argmax(dim=-1).tolist()
                    # convert index to labels
                    mdata["logits"] = label_logit.softmax(dim=-1).tolist()
                    mdata["predictions"] = [self.label_lists[p] for p in preds]
                    mdata["prediction"] = mdata["predictions"][-1]
                    avg_ice_num += mdata["ice_num"]

                    if self.visualizer.contains(i):
                        tokens = self.dataset_reader.tokenizer.convert_ids_to_tokens(
                            entry["input_ids"][0], skip_special_tokens=False
                        )
                        # take care of space tokens
                        space_tok = self.dataset_reader.tokenizer.tokenize(" ")[0]
                        tokens = [t.replace(space_tok, "") for t in tokens]
                        attns = torch.cat(outputs.attentions, dim=0)
                        self.visualizer.visualize(i, tokens, attns[:, :, label_positions[-1], :])

            else:
                outputs = self.model.generate(
                    input_ids=entry.input_ids,  # TODO, check here because no_special_token is added
                    attention_mask=entry.attention_mask,
                    eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
                    pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                    do_sample=False,  # always use greedy decode here
                    **self.generation_kwargs,
                )
                prompt_len = int(entry.attention_mask.shape[1])
                for mdata, output in zip(metadata, outputs.tolist()):
                    generated = self.dataset_reader.tokenizer.decode(output[prompt_len:])
                    mdata["generated"] = generated.strip(self.dataset_reader.tokenizer.pad_token).strip()
                    avg_ice_num += len(mdata["ice_prompts_list"])

            res.extend(metadata)

        save_json(f"{self.output_file}tmp_{self.accelerator.device}.bin", res)

        logger.info(f"Average number of in-context examples after truncating is {avg_ice_num / len(res):.4f}")

    def inference(self):
        self.forward()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            data = []
            for path in glob.glob(f"{self.output_file}tmp_*.bin"):
                with open(path) as f:
                    data.extend(json.load(f))

            if hasattr(self.evaluator, "post_process"):
                preds, refs = self.evaluator.post_process(data)
            else:
                preds = [i["prediction"] for i in data]
                refs = [i["label"] for i in data]
            metric = self.evaluator.compute(references=refs, predictions=preds)
            logger.info(f"metric: {metric}")

            results = "-".join([f"{key}-{val:.3g}" for key, val in metric.items()])
            basename, extension = self.output_file.rsplit(".", maxsplit=1)

            output_file = f"{basename}-{results}.{extension}"
            save_json(output_file, data, indent=2)

            for path in glob.glob(f"{self.output_file}tmp_*.bin"):
                os.remove(path)
