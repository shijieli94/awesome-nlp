import numpy as np


class InferenceDatasetReader:
    def __init__(
        self,
        dataset_reader,
        index_reader,
        ntokens,
        num_ice=None,
        ice_separator="\n",
        move_nearest_to_the_end=True,
        **kwargs,
    ):
        self.ntokens = ntokens
        self.num_ice = num_ice
        self.ice_separator = ice_separator
        self.move_nearest_to_the_end = move_nearest_to_the_end

        self.index_reader = index_reader
        self.dataset_wrapper = dataset_reader.dataset_wrapper
        self.encoded_dataset = dataset_reader.encoded_dataset

        self.tokenizer = dataset_reader.tokenizer
        # make sure the tokenizer is left-padded, so that we can simply use the last output logits
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # left-pad will cause some problem we use right pad
        self.tokenizer.padding_side = "right"

    def __len__(self):
        return len(self.dataset_wrapper)

    def __getitem__(self, index):
        entry = self.dataset_wrapper[index]
        prompt_text = self.encoded_dataset[index]["metadata"]["text"]
        prompt_len = self.encoded_dataset[index]["metadata"]["len"]

        ice_ctx, ice_ctx_label = self.get_ice_prompt(entry, prompt_len)
        ice_text = [i["metadata"]["text"] for i in ice_ctx]
        ice_lens = [i["metadata"]["len"] for i in ice_ctx]

        ice_prompt = "" if len(ice_ctx) == 0 else self.ice_separator.join(ice_text) + self.ice_separator

        entry["prompt"] = ice_prompt + prompt_text + self.dataset_wrapper.gen_prefix
        entry["ice_num"] = len(ice_text)

        # because of the ice_seperator, so after tokenize, length will exceed n_tokens_in_prompt
        tokenized_example = self.tokenizer(
            entry["prompt"], truncation=False, return_tensors="pt", add_special_tokens=False
        )

        tokenized_separator = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.ice_separator))
        assert len(tokenized_separator) == 1

        # ice examples have added a label token, so the last token is the label, for this reason,
        # (len - 1) is the index of label as input, so (len - 2) is the index of label as output
        # since seperator is used as concatenation, so we should shift one position for following examples.
        prompt_labels = np.cumsum(ice_lens) + np.array(list(range(-2, len(ice_lens) - 2)))

        # prompt doesn't include a label, so the index of last token as input is the index of label as output
        # prompt_labels[-1] is the index of label as output, plus 2 is the index of the first token of prompt
        # as the input, further add prompt_len is the index of the last token of prompt as input.
        entry["labels"] = ice_ctx_label + [entry["label"]]
        if len(prompt_labels) == 0:
            entry["label_positions"] = [prompt_len - 1]
        else:
            entry["label_positions"] = prompt_labels.tolist() + [int(prompt_labels[-1]) + 2 + prompt_len]

        return {
            "input_ids": tokenized_example.input_ids[0],
            "attention_mask": tokenized_example.attention_mask[0],
            "metadata": entry,
        }

    def get_ice_prompt(self, entry, prompt_len):
        if "ctxs" not in entry or self.num_ice == 0:
            return [], []

        ctx = [self.index_reader[i] for i in entry["ctxs"]]
        ctx_label = [self.index_reader.dataset_wrapper[i]["label"] for i in entry["ctxs"]]

        if self.num_ice is not None:
            ctx = ctx[: self.num_ice]
            ctx_label = ctx_label[: self.num_ice]

        max_prompts = np.searchsorted(np.cumsum([i["metadata"]["len"] for i in ctx]), self.ntokens - prompt_len)
        ctx = ctx[:max_prompts]
        ctx_label = ctx_label[:max_prompts]
        if self.move_nearest_to_the_end:
            ctx = ctx[::-1]  # more similar more close
            ctx_label = ctx_label[::-1]

        return ctx, ctx_label

    def shard(self, accelerator):
        self.dataset_wrapper.dataset = self.dataset_wrapper.dataset.shard(
            num_shards=accelerator.num_processes, index=accelerator.process_index
        )
        self.encoded_dataset = self.encoded_dataset.shard(
            num_shards=accelerator.num_processes, index=accelerator.process_index
        )
        assert len(self.dataset_wrapper.dataset) == len(self.encoded_dataset)
