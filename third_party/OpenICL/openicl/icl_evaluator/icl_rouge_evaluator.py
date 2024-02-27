"""ROUGE evaluator"""
from openicl.icl_evaluator import BaseEvaluator


class RougeEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__(metric="rouge")

    def score(self, predictions, references, **kwargs):
        assert len(predictions) == len(references)
        scores = self.metric.compute(predictions=predictions, references=references, **kwargs)
        return scores
