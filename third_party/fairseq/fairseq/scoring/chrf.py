# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer


@dataclass
class ChrFScorerConfig(FairseqDataclass):
    chrf_char_order: int = field(default=6, metadata={"help": "character n-gram order"})
    chrf_word_order: int = field(
        default=0, metadata={"help": "word n-gram order. If equals to 2, the metric is referred to as chrF++"}
    )


@register_scorer("chrf", dataclass=ChrFScorerConfig)
class ChrFScorer(BaseScorer):
    def __init__(self, args):
        super(ChrFScorer, self).__init__(args)
        import sacrebleu

        self.sacrebleu = sacrebleu
        self.char_order = args.chrf_char_order
        self.word_order = args.chrf_word_order

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.pred.append(pred)

    def score(self, order=4):
        return self.result_string(order).score

    def result_string(self, order=4):
        if order != 4:
            raise NotImplementedError
        return self.sacrebleu.corpus_chrf(
            self.pred, [self.ref], char_order=self.char_order, word_order=self.word_order
        ).format()
