# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from fairseq import utils
from fairseq.data import data_utils, encoders, indexed_dataset
from fairseq.data.append_token_dataset import AppendTokenDataset
from fairseq.data.concat_dataset import ConcatDataset
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.data.prepend_token_dataset import PrependTokenDataset
from fairseq.data.shorten_dataset import TruncateDataset
from fairseq.data.strip_token_dataset import StripTokenDataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.logging import metrics
from fairseq.tasks import FairseqTask, register_task
from omegaconf import II

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos_src=False,
    prepend_bos_tgt=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    dataset_class=None,
    **unused,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError("Dataset not found: {} ({})".format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info("{} {} {}-{} {} examples".format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos_src:
        assert hasattr(src_dict, "bos_index")
        logger.info(f"prepending bos index ({src_dict.bos()}) to the source dictionary")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())

    if tgt_dataset is not None and prepend_bos_tgt:
        assert hasattr(tgt_dict, "bos_index")
        logger.info(f"prepending bos index ({tgt_dict.bos()}) to the target dictionary")
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index("[{}]".format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index("[{}]".format(tgt)))
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    dataset_class = dataset_class or LanguagePairDataset
    return dataset_class(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class TranslationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(default=False, metadata={"help": "load the binarized alignments"})
    left_pad_source: bool = field(default=True, metadata={"help": "pad the source on the left"})
    left_pad_target: bool = field(default=False, metadata={"help": "pad the target on the left"})
    max_source_positions: int = field(default=1024, metadata={"help": "max number of tokens in the source sequence"})
    max_target_positions: int = field(default=1024, metadata={"help": "max number of tokens in the target sequence"})
    upsample_primary: int = field(default=-1, metadata={"help": "the amount of upsample primary dataset"})
    truncate_source: bool = field(default=False, metadata={"help": "truncate source to max-source-positions"})
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II("dataset.dataset_impl")
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(default=False, metadata={"help": "evaluation with BLEU scores"})
    eval_bleu_order: int = field(default=EVAL_BLEU_ORDER, metadata={"help": "the order of BLEU"})
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={"help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'},
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"})
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    extra_special_tokens: int = field(
        default=0,
        metadata={
            "help": "number of special tokens to add after data preprocessing. "
            "This is useful because the preprocessing only contains four default special tokens. "
            "Passing extra_special_symbols will shift token ids. "
            "So in this case, special tokens will be appended to the last."
        },
    )
    decode_remove_special_tokens: bool = field(
        default=False, metadata={"help": "remove appended special tokens when decoding"}
    )
    filter_min_sizes: List[int] = field(
        default_factory=lambda: [],
        metadata={"help": "filter out samples that do not satisfy the specified min size constraints."},
    )
    filter_max_sizes: List[int] = field(
        default_factory=lambda: [],
        metadata={"help": "filter out samples that do not satisfy the specified max size constraints."},
    )
    filter_ratios: List[float] = field(
        default_factory=lambda: [],
        metadata={
            "help": "filter out samples that do not satisfy the specified ratio constraints."
            "If one value is provided, then tgt > src * filter_ratios[0] and src > tgt * filter_ratios[0] will be filtered out, "
            "If two values are provided, then tgt > src * filter_ratios[0] and src > tgt * filter_ratios[1] will be filtered out."
        },
    )


@register_task("translation", dataclass=TranslationConfig)
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationConfig

    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception("Could not infer language pair, please provide it explicitly")

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang)))
        if cfg.extra_special_tokens > 0:
            logger.info(
                "Appending {} extra special tokens to the source and target dictionaries".format(
                    cfg.extra_special_tokens
                )
            )
            src_dict.append_extra_tokens(n=cfg.extra_special_tokens)
            tgt_dict.append_extra_tokens(n=cfg.extra_special_tokens)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            **kwargs,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_bleu:
            import sacrebleu

            self.bleu = sacrebleu.BLEU(
                tokenize="none" if self.cfg.eval_tokenized_bleu else None,
                max_ngram_order=self.cfg.eval_bleu_order,
                trg_lang=self.cfg.target_lang,  # useful for deciding best tokenizer for languages such as Chinese and Japanese
            )

            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args))

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=int(meters["_bleu_sys_len"].sum),
                        ref_len=int(meters["_bleu_ref_len"].sum),
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        max_source_positions = self.cfg.max_source_positions
        max_target_positions = self.cfg.max_target_positions

        if len(self.cfg.filter_max_sizes) == 0:
            return max_source_positions, max_target_positions
        if len(self.cfg.filter_max_sizes) == 1:
            return (
                min(max_source_positions, self.cfg.filter_max_sizes[0]),
                min(max_target_positions, self.cfg.filter_max_sizes[0]),
            )
        if len(self.cfg.filter_max_sizes) == 2:
            return (
                min(max_source_positions, self.cfg.filter_max_sizes[0]),
                min(max_target_positions, self.cfg.filter_max_sizes[1]),
            )
        raise ValueError(f"Max length of `--filter-max-sizes` is 2 but found {len(self.cfg.filter_max_sizes)}")

    def min_positions(self):
        """Return the min sentence length allowed by the task."""
        if len(self.cfg.filter_min_sizes) == 0:
            return None, None
        if len(self.cfg.filter_min_sizes) == 1:
            return self.cfg.filter_min_sizes[0], self.cfg.filter_min_sizes[0]
        if len(self.cfg.filter_min_sizes) == 2:
            return self.cfg.filter_min_sizes[0], self.cfg.filter_min_sizes[1]
        raise ValueError(f"Max length of `--filter-min-sizes` is 2 but found {len(self.cfg.filter_min_sizes)}")

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        def decode(toks, unk_string=None):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=unk_string,
                extra_symbols_to_ignore=self.tgt_dict.extra() if self.cfg.decode_remove_special_tokens else None,
            )
            if self.tokenizer and not self.cfg.eval_tokenized_bleu:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"], unk_string="UNKNOWNTOKENINHYP"))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    unk_string="UNKNOWNTOKENINREF",  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])

        return self.bleu.corpus_score(hyps, [refs])

    def filter_indices_by_size(self, indices, dataset, max_positions=None, ignore_invalid_inputs=False):
        # filter indices by max_positions
        indices, ignored = dataset.filter_indices_by_size(indices, max_positions)

        # filter indices by min_positions
        min_src_size, min_tgt_size = self.min_positions()

        ignored_idx = np.array([False] * len(indices))
        if min_src_size is not None:
            ignored_idx |= dataset.src_sizes[indices] < min_src_size
        if dataset.tgt_sizes is not None and min_tgt_size is not None:
            ignored_idx |= dataset.tgt_sizes[indices] < min_tgt_size

        ignored += indices[ignored_idx].tolist()
        indices = indices[~ignored_idx]

        # filter indices by ratio
        if len(self.cfg.filter_ratios) > 0:
            if len(self.cfg.filter_ratios) == 1:
                upscale_src = upscale_tgt = self.cfg.filter_ratios[0]
            elif len(self.cfg.filter_ratios) == 2:
                upscale_src, upscale_tgt = self.cfg.filter_ratios
            else:
                raise ValueError(f"filter_ratios should have length 1 or 2, but got {len(self.cfg.filter_ratios)}")

            ignored_idx = (dataset.tgt_sizes[indices] > dataset.src_sizes[indices] * upscale_src) | (
                dataset.src_sizes[indices] > dataset.tgt_sizes[indices] * upscale_tgt
            )
            ignored += indices[ignored_idx].tolist()
            indices = indices[~ignored_idx]

        assert len(ignored) + len(indices) == len(dataset)

        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception(
                    (
                        "Size of sample #{} is invalid (={}) since min_positions={}, max_positions={}, max_ratio={}, "
                        "skip this example with --skip-invalid-size-inputs-valid-test"
                    ).format(
                        ignored[0],
                        dataset.size(ignored[0]),
                        self.min_positions(),
                        max_positions,
                        self.cfg.filter_ratios,
                    )
                )
            logger.warning(
                (
                    "{:,} samples have invalid sizes and will be skipped, "
                    "min_positions={}, max_positions={}, max_ratio={}, first few sample ids={}"
                ).format(len(ignored), self.min_positions(), max_positions, self.cfg.filter_ratios, ignored[:10])
            )
        return indices
