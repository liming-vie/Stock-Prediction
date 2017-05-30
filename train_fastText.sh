#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

if [ $# != 5]; then
  echo 'Usage: sh train_fastText.sh stock_dir news_dir embed_train_file result_dir vector_file'
  exit 1
fi

PRICE_DIR=$1
NEWS_DIR=$2
EMBED_TRAIN_FILE=$3
RESULTDIR=$4
VECTOR_FILE=$5
BINDIR=../fastText


$BINDIR/fasttext supervised -input $TRAIN_FILE -output "${RESULTDIR}/model" -dim 256 -lr 0.1 -wordNgrams 2 -minCount 5 -bucket 10000000 -epoch 5 -thread 24

$BINDIR/fasttext print-sentence-vectors $RESULTDIR/model.bin < $EMBED_TRAIN_FILE > $RESULTDIR/VECTOR_FILE
