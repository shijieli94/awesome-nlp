"""Base Evaluator"""
import evaluate


class BaseEvaluator:
    def __init__(self, metric) -> None:
        self.metric = evaluate.load(metric)

    def score(self, predictions, references, **kwargs):
        raise NotImplementedError("Method hasn't been implemented yet")
