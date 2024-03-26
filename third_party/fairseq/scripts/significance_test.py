"""
See `"Statistical Significance Tests for Machine Translation Evaluation"
(Koehn et al., 2014) <https://aclanthology.org/W04-3250/>`_.
"""

import argparse
import logging
import os
import re
import sys
from collections import defaultdict

from sacrebleu import BLEU
from sacrebleu.significance import PairedTest

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger("awesome_nlp")


def parse_args():
    parser = argparse.ArgumentParser(description="statistical significance test")
    # fmt: off
    parser.add_argument("--baseline", type=str, required=True,
                        help="output file name from baseline")
    parser.add_argument("--systems", type=str, required=True, nargs="+",
                        help="output file names from systems")
    parser.add_argument("--filename-pattern", type=str, default=None,
                        help="name suffix of checkpoint files")
    parser.add_argument("--test-type", type=str, default="bs",
                        choices=("bs", "ar"),
                        help="type of significance test")
    parser.add_argument("--trg-lang", type=str, default=None,
                        help="target language used for deciding the best tokenize.")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="number of resampling set size")
    parser.add_argument("--metrics", type=str, nargs="+", help="metrics to compute")
    # fmt: on
    return parser.parse_args()


def cli_main():
    args = parse_args()

    statics = defaultdict(lambda: defaultdict(list))

    def _format_filename(filename):
        return filename if args.filename_pattern is None else args.filename_pattern.format(filename)

    def _split_line(line, maxsplit=1):
        return line.strip().split(maxsplit=maxsplit)[-1]

    def _parse_generated_file(name, file):
        assert name not in statics, f"Found duplicate system name {name}"

        with open(file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if re.search(r"^T", line):
                    statics[name]["T"].append(_split_line(line, maxsplit=1))
                elif re.search(r"^DT", line):
                    statics[name]["DT"].append(_split_line(line, maxsplit=1))
                elif re.search(r"^H", line):
                    statics[name]["H"].append(_split_line(line, maxsplit=2))
                elif re.search(r"^DH", line):
                    statics[name]["DH"].append(_split_line(line, maxsplit=2))

    _parse_generated_file(args.baseline, _format_filename(args.baseline))

    named_systems_status = [args.baseline]
    for system in args.systems:
        _parse_generated_file(system, _format_filename(system))
        named_systems_status.append(system)

    def _headline(msg, width=40):
        width -= len(msg) // 2
        return "-" * width + f" {msg} " + "-" * width

    scores = {}
    if "sacrebleu" in args.metrics:
        assert args.trg_lang is not None, "trg_lang is required for sacrebleu"
        logger.info("-" * 20 + " SACREBLEU " + "-" * 20)
        metric = BLEU(references=[statics[args.baseline]["DT"]], tokenize=None, trg_lang=args.trg_lang)

        named_systems = []
        for system in named_systems_status:
            named_systems.append((system, statics[system]["DH"]))

        score = PairedTest(
            named_systems,
            metrics={"sacreBLEU": metric},
            references=None,
            test_type=args.test_type,
            n_samples=args.n_samples,
        )
        scores["sacreBLEU"] = score()

    if "tokenized_bleu" in args.metrics:
        logger.info(_headline("-" * 20 + " TOKENIZED BLEU " + "-" * 20))
        metric = BLEU(references=[statics[args.baseline]["T"]], tokenize="none")

        named_systems = []
        for system in named_systems_status:
            named_systems.append((system, statics[system]["H"]))

        score = PairedTest(
            named_systems,
            metrics={"tokenized-BLEU": metric},
            references=None,
            test_type=args.test_type,
            n_samples=args.n_samples,
        )
        scores["tokBLEU"] = score()

    logger.info("*" * 30 + " FINAL RESULTS " + "*" * 30)
    for name, (signature, results) in scores.items():
        logger.info("-" * 20 + f"{name}: {list(signature.values())[0]}" + "-" * 20)
        for system, result in zip(*results.values()):
            if result.p_value is None:
                p_value = None
            elif result.p_value < 0.001:
                p_value = "{:.6f} < 0.001".format(result.p_value)
            elif result.p_value < 0.01:
                p_value = "{:.6f} < 0.01".format(result.p_value)
            elif result.p_value < 0.05:
                p_value = "{:.6f} < 0.05".format(result.p_value)
            else:
                p_value = "{:.6f} > 0.05".format(result.p_value)

            logger.info(
                "{}: {:.2f} | mean: {:.2f} | ci: {:.2f} | p-value: {}".format(
                    system, result.score, result.mean, result.ci, p_value
                )
            )


if __name__ == "__main__":
    cli_main()
