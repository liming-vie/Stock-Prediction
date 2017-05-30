#!/bin/bash

if [ $# != 2 ]; then
  echo 'Usage: sh train_word2vec.sh corpus_file vector_file'
  exit 1
fi

BIN_DIR=../word2vec/bin

TEXT_DATA=$1
VECTOR_DATA=$2

time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 128 -window 10 -negative 5 -hs 1 -sample 1e-4 -threads 24 -binary 0
