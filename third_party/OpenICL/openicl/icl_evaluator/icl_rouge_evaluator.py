"""ROUGE evaluator"""

import evaluate
from openicl.icl_evaluator import BaseEvaluator


class RougeEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        metric = evaluate.load("rouge")
        scores = metric.compute(predictions=predictions, references=references)
        return scores
