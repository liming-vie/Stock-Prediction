#!/bin/bash
news_dir='../data/CnNewsReport'
segment_dir='../output/segmented'
vocab_file='../output/vocab'
token_ids_dir='../output/token_ids'
polar_seed_file='../output/porlar_seed'
embedding_train_file='../output/embedding_train_corpus'
K=20

python data_utils.py $news_dir $segment_dir $vocab_file $token_ids_dir $embedding_train_file
python pmi.py $vocab_file $token_ids_dir $polar_seed_file $K

glove_embedding='../output/glove/vectors.bin'
word2vec_embedding='../output/word2vec/vectors.bin'
sh train_glove.sh &
sh train_word2vec.sh &
wait
