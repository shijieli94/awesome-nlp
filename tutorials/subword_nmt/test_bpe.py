#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import codecs
import inspect
import os
import sys
import unittest

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from apply_bpe import BPE
from get_vocab import get_vocab
from learn_bpe import learn_bpe


class TestBPELearnMethod(unittest.TestCase):
    def test_learn_bpe(self):
        infile = codecs.open(os.path.join(current_dir, "data", "corpus.en"), encoding="utf-8")
        outfile = codecs.open(os.path.join(current_dir, "data", "bpe.out"), "w", encoding="utf-8")
        learn_bpe(infile, outfile, 1000)
        infile.close()
        outfile.close()

        outlines = open(os.path.join(current_dir, "data", "bpe.out"))
        reflines = open(os.path.join(current_dir, "data", "bpe.ref"))

        for line, line2 in zip(outlines, reflines):
            self.assertEqual(line, line2)

        outlines.close()
        reflines.close()


class TestBPESegmentMethod(unittest.TestCase):
    def setUp(self):
        with codecs.open(os.path.join(current_dir, "data", "bpe.ref"), encoding="utf-8") as bpefile:
            self.bpe = BPE(bpefile)

        self.infile = codecs.open(os.path.join(current_dir, "data", "corpus.en"), encoding="utf-8")
        self.reffile = codecs.open(os.path.join(current_dir, "data", "corpus.bpe.ref.en"), encoding="utf-8")

    def tearDown(self):
        self.infile.close()
        self.reffile.close()

    def test_apply_bpe(self):
        for line, ref in zip(self.infile, self.reffile):
            out = self.bpe.process_line(line)
            self.assertEqual(out, ref)

    def test_trailing_whitespace(self):
        """BPE.proces_line() preserves leading and trailing whitespace"""

        orig = "  iron cement  \n"
        exp = "  ir@@ on c@@ ement  \n"

        out = self.bpe.process_line(orig)
        self.assertEqual(out, exp)

    def test_utf8_whitespace(self):
        """UTF-8 whitespace is treated as normal character, not word boundary"""

        orig = "iron\xa0cement\n"
        exp = "ir@@ on@@ \xa0@@ c@@ ement\n"

        out = self.bpe.process_line(orig)
        self.assertEqual(out, exp)

    def test_empty_line(self):
        orig = "\n"
        exp = "\n"

        out = self.bpe.process_line(orig)
        self.assertEqual(out, exp)


class TestGetVocabMethod(unittest.TestCase):
    def test_get_vocab(self):
        infile = codecs.open(os.path.join(current_dir, "data", "corpus.en"), encoding="utf-8")
        outfile = codecs.open(os.path.join(current_dir, "data", "vocab.out"), "w", encoding="utf-8")
        get_vocab(infile, outfile)
        infile.close()
        outfile.close()

        outlines = open(os.path.join(current_dir, "data", "vocab.out"))
        reflines = open(os.path.join(current_dir, "data", "vocab.ref"))

        for line, line2 in zip(outlines, reflines):
            self.assertEqual(line, line2)

        outlines.close()
        reflines.close()

    def test_segment_char_ngrams(self):
        from segment_char_ngrams import create_parser, segment_char_ngrams

        parser = create_parser()
        args = parser.parse_args(
            # fmt: off
            [
                "--vocab", os.path.join(current_dir, "data", "vocab.ref"),
                "--input", os.path.join(current_dir, "data", "corpus.en"),
                "--output", os.path.join(current_dir, "data", "ngram.out"),
                "-n", "2",
                "--shortlist", "3000",
            ]
            # fmt: on
        )

        if sys.version_info < (3, 0):
            args.separator = args.separator.decode("UTF-8")

        segment_char_ngrams(args)

        args.vocab.close()
        args.input.close()
        args.output.close()


if __name__ == "__main__":
    unittest.main()
