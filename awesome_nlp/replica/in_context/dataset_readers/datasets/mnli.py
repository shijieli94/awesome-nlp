from . import BaseDataset, Fields, register_dataset


def get_q(entry):
    return entry["premise"], entry["hypothesis"]


def get_a(entry):
    return get_labels()[entry["label"]]


def get_p(entry):
    return f"{entry['premise']} Can we say \"{entry['hypothesis']}\"? {get_a(entry)}"


def get_gen(entry):
    # hypothesis, premise = get_q(entry)
    return f"{entry['premise']} Can we say \"{entry['hypothesis']}\"?"


def get_labels():
    return {0: "Yes", 1: "Maybe", 2: "No"}


@register_dataset()
class DatasetWrapper(BaseDataset):
    name = "mnli"
    question_field = ["hypothesis", "premise"]
    answer_field = "label"
    hf_dataset = "LysandreJik/glue-mnli-train"
    hf_dataset_name = "glue-mnli-train"
    metric = "accuracy"
    fields = Fields({"q": get_q, "a": get_a, "p": get_p, "gen": get_gen, "labels": get_labels})
