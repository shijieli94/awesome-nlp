#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Usage:
# e.g.
# bash prepare-bpe.sh data/iwslt17 exp_test

if [ -d "mosesdecoder" ]; then
    echo "mosesdecoder already exists, skipping download"
else
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
fi

if [ -d "subword-nmt" ]; then
    echo "subword-nmt already exists, skipping download"
else
    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone https://github.com/rsennrich/subword-nmt.git
fi

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
TC=$SCRIPTS/recaser/truecase.perl
TRAINTC=$SCRIPTS/recaser/train-truecaser.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=30000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

data=$1
prep=$2/tokenized.en-de
src=en
tgt=de
lang=$src-$tgt
tmp=$prep/tmp

BPE_CODE=$prep/code
if [ -f "$BPE_CODE" ]; then
    echo "BPE code $BPE_CODE is already exist, skipping data preparation."
    exit
fi

mkdir -p $tmp $prep

echo "filter out empty lines from original data and split doc with empty line..."
for D in train dev test; do
    sf=$data/concatenated_${src}2${tgt}_${D}_${src}.txt
    tf=$data/concatenated_${src}2${tgt}_${D}_${tgt}.txt
    RD=$D
    if [ $D == "dev" ]; then
        RD=valid
    fi

    rf=$tmp/$RD.$lang.tag
    echo $rf

    paste -d"\t" $sf $tf | \
    grep -v -P "^\s*\t" | \
    grep -v -P "\t\s*$" | \
    sed -e 's/\r//g' > $rf

    cut -f 1 $rf | \
    sed -e 's/^<d>\s*$//g' | \
    perl $TOKENIZER -threads 8 -l $src > $tmp/$RD.$src

    cut -f 2 $rf | \
    sed -e 's/^<d>\s*$//g' | \
    perl $TOKENIZER -threads 8 -l $tgt > $tmp/$RD.$tgt
done

echo "truecase the train/valid/test data..."
TCM_SRC=$tmp/truecase.$src.mdl
echo $TCM_SRC
$TRAINTC --model $TCM_SRC --corpus $tmp/train.$src

TCM_TGT=$tmp/truecase.$tgt.mdl
echo $TCM_TGT
$TRAINTC --model $TCM_TGT --corpus $tmp/train.$tgt

for D in train valid test; do
    $TC --model $TCM_SRC < $tmp/$D.$src > $tmp/$D.$src.tc
    echo $tmp/$D.$src.tc
    $TC --model $TCM_TGT < $tmp/$D.$tgt > $tmp/$D.$tgt.tc
    echo $tmp/$D.$tgt.tc
done

echo "learn_bpe.py on ${TRAIN}..."
TRAIN=$tmp/train.$lang.tc
cat $tmp/train.$src.tc > $TRAIN
cat $tmp/train.$tgt.tc >> $TRAIN

python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for F in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${F}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$F.tc > $tmp/$F.tc.bpe
    done
done

echo "apply doc-level special tags..."
for L in $src $tgt; do
    for F in train.$L valid.$L test.$L; do
        cat $tmp/$F.tc.bpe | \
        # replace empty line with [DOC]
        sed -e 's/^$/[DOC]/g' | \
        # connect all lines into one line
        sed -z -e 's/\n/ [SEP] /g' | \
        # replace the begin of doc with newline
        sed -e 's/ \[DOC\] \[SEP\] /\n/g' | \
        # handle the begin-symbol of the first doc
        sed -e 's/\[DOC\] \[SEP\] //g' | \
        # replace all [SEP] with </s>
        sed -e 's/\[SEP\]/<\/s>/g' > $prep/$F
    done
done

tok_path=$2/tokenized.en-de
seg_path=$2/segmented.en-de

python prepare_doc_mt.py --datadir $tok_path --destdir $seg_path/doc/ --source-lang $src --target-lang $tgt --max-tokens 512 --max-sents 1000
python prepare_doc_mt.py --datadir $tok_path --destdir $seg_path/sent/ --source-lang $src --target-lang $tgt --max-tokens 512 --max-sents 1
