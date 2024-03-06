from . import BaseDataset, Fields, register_dataset


def get_q(entry):
    return entry["text"]


def get_a(entry):
    return get_labels()[entry["label"]]


def get_p(entry):
    return f"{get_q(entry)} It is {get_a(entry)}"


def get_gen(entry):
    return f"{get_q(entry)} It is"


def get_labels():
    return {0: "terrible", 1: "bad", 2: "OK", 3: "good", 4: "great"}


@register_dataset()
class DatasetWrapper(BaseDataset):
    name = "sst5"
    question_field = ["text"]
    answer_field = "label"
    hf_path = "SetFit/sst5"
    hf_name = None
    metric = "accuracy"
    fields = Fields({"q": get_q, "a": get_a, "p": get_p, "gen": get_gen, "labels": get_labels})
