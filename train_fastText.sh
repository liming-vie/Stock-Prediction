#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

if [ $# != 4 ]; then
  echo 'Usage: sh train_fastText.sh train_file test_file output_dir vector_file'
  exit 1
fi

TRAIN_FILE=$1
TEST_FILE=$2
RESULTDIR=$3
VECTOR_FILE=$4
EMBED_FILE=$5
DIM=300
BINDIR=../fastText

mkdir -p $RESULTDIR

$BINDIR/fasttext supervised -input $TRAIN_FILE -output "${RESULTDIR}/model" -dim $DIM -lr 0.1 -wordNgrams 3 -minCount 5 -bucket 10000000 -epoch 50 -thread 24

$BINDIR/fasttext print-sentence-vectors $RESULTDIR/model.bin < $TEST_FILE > $RESULTDIR/$VECTOR_FILE