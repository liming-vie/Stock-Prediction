#!/bin/bash

BIN_DIR=../word2vec/bin

TEXT_DATA=../output/embedding_train_corpus
VECTOR_DATA=../output/word2vec/vectors.bin


time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 256 -window 10 -negative 5 -hs 1 -sample 1e-4 -threads 10 -binary 1
