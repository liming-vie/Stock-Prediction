#!/bin/bash

if [ $# != 3 ]; then
  echo 'Usage: sh train_word2vec.sh corpus_file output_dir vector_file'
  exit 1
fi

BIN_DIR=../word2vec/bin

TEXT_DATA=$1
OUTPUT_DIR=$2
VECTOR_FILE=$OUTPUT_DIR/$3
VOCAB_FILE=$OUTPUT_DIR/vocab.txt

mkdir -p $OUTPUT_DIR

THREAD_NUM=24
DIM=256

time $BIN_DIR/word2vec    \
  -train $TEXT_DATA       \
  -output $VECTOR_FILE    \
  -cbow 0 -size $DIM      \
  -window 10              \
  -negative 5             \
  -hs 1                   \
  -sample 1e-4            \
  -threads $THREAD_NUM    \
  -binary 0               \
  -save-vocab $VOCAB_FILE
