from . import BaseDataset, Fields, register_dataset


def get_q(entry):
    return entry["sentence1"], entry["sentence2"]


def get_a(entry):
    return get_labels()[entry["label"]]


def get_p(entry):
    return f"{entry['sentence1']} Can we say \"{entry['sentence2']}\"? {get_a(entry)}"


def get_gen(entry):
    return f"{entry['sentence1']} Can we say \"{entry['sentence2']}\"?"


def get_labels():
    return {0: "No", 1: "Yes"}


@register_dataset()
class DatasetWrapper(BaseDataset):
    name = "mrpc"
    question_field = ["sentence1", "sentence2"]
    answer_field = "label"
    hf_path = "glue"
    hf_name = "mrpc"
    metric = "accuracy"
    fields = Fields({"q": get_q, "a": get_a, "p": get_p, "gen": get_gen, "labels": get_labels})
