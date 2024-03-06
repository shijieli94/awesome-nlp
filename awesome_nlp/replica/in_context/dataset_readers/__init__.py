import pandas as pd


def _encode_field(example, idx, **kwargs):
    text = kwargs["field_getter"](example)
    if isinstance(text, str):
        text = [text]
    # set truncation to False so that len is the real len
    tokenized_inputs = kwargs["tokenizer"](
        *text, truncation=False, return_tensors="pt", add_special_tokens=kwargs["add_special_tokens"]
    )
    return {
        "input_ids": tokenized_inputs.input_ids[0],
        "attention_mask": tokenized_inputs.attention_mask[0],
        "metadata": {"id": idx, "len": len(tokenized_inputs.input_ids[0]), "text": " ".join(text)},
    }


def encode_field(tokenizer, dataset_wrapper, field, add_special_tokens=True, num_proc=4):
    remove_columns = [col for col in dataset_wrapper.dataset.column_names]
    encoded_dataset = dataset_wrapper.dataset.map(
        _encode_field,
        load_from_cache_file=False,
        with_indices=True,
        remove_columns=remove_columns,
        num_proc=num_proc,
        fn_kwargs={
            "field_getter": dataset_wrapper.fields[field],
            "tokenizer": tokenizer,
            "add_special_tokens": add_special_tokens,
        },
    )
    return encoded_dataset


def deduplicate(encoded_dataset):
    df = pd.DataFrame(encoded_dataset)
    df["uid"] = df["input_ids"].astype(str)
    is_dup = df.duplicated(subset=["uid"], keep="first")
    return is_dup[~is_dup].index.values if any(is_dup) else None
