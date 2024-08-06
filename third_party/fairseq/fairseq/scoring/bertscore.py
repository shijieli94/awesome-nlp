# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import numpy as np
from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer


@dataclass
class BertScoreScorerConfig(FairseqDataclass):
    bert_score_lang: str = field(default="en", metadata={"help": "BERTScore language"})
    bert_score_type: str = field(
        default="F", metadata={"help": "Recall or Precision or F1 score of bert_score", "choices": ["P", "R", "F"]}
    )
    bert_score_rescale: bool = field(default=True, metadata={"help": "Rescale BERTScore to the range [0, 1]"})


@register_scorer("bert_score", dataclass=BertScoreScorerConfig)
class BertScoreScorer(BaseScorer):
    def __init__(self, cfg):
        super(BertScoreScorer, self).__init__(cfg)
        try:
            import bert_score as _bert_score
        except ImportError:
            raise ImportError("Please install BERTScore: pip install bert-score")

        self.cfg = cfg
        self._bert_score = _bert_score

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.pred.append(pred)

    def score(self, order=4, return_hash=False):
        (P, R, F), hash_ = self._bert_score.score(
            self.pred,
            self.ref,
            lang=self.cfg.bert_score_lang,
            rescale_with_baseline=self.cfg.bert_score_rescale,
            return_hash=True,
        )
        score = {"P": P, "R": R, "F": F}[self.cfg.bert_score_type]
        if return_hash:
            return score, hash_
        return score

    def result_string(self, order=4):
        score, hash_ = self.score(return_hash=True)
        score = np.mean(score.numpy())
        return f"BERTScore-{self.cfg.bert_score_type}: {score:.4f} | {hash_}"
