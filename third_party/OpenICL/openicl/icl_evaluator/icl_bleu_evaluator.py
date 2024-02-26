"""BLEU evaluator"""
from typing import Dict, List

import evaluate
from openicl.icl_evaluator import BaseEvaluator


class BleuEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        metric = evaluate.load("sacrebleu")
        scores = metric.compute(predictions=predictions, references=references)
        return scores
