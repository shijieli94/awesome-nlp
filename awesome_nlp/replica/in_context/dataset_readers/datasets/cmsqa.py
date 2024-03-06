import evaluate

from . import BaseDataset, Fields, register_dataset


class Metric(object):
    def __init__(self, cache_dir):
        self.metric = evaluate.load("accuracy", cache_dir=cache_dir)

    def post_process(self, preds, refs):
        refs = [ord(ref["answerKey"]) - ord("A") for ref in refs]
        return preds, refs

    def compute(self, predictions, references):
        return self.metric.compute(references=references, predictions=predictions)


def get_q(entry):
    return entry["question"]


def get_a(entry):
    return get_labels(entry)[ord(entry["answerKey"]) - ord("A")]


def get_p(entry):
    return f"Q: {get_q(entry)}\tA: {get_a(entry)}"


def get_gen(entry):
    return f"Q: {get_q(entry)}\tA:"


def get_labels(entry):
    return entry["choices"]["text"]


@register_dataset()
class DatasetWrapper(BaseDataset):
    name = "cmsqa"
    question_field = ["question"]
    answer_field = "answerKey"
    hf_dataset = "commonsense_qa"
    hf_dataset_name = None
    metric = Metric
    fields = Fields({"q": get_q, "a": get_a, "p": get_p, "gen": get_gen, "labels": get_labels})
