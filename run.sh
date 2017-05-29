#!/bin/bash
news_dir='../data/CnNewsReport'
segment_dir='../output/segmented'
vocab_file='../output/vocab'
token_ids_dir='../output/token_ids'
polar_seed_file='../output/porlar_seed'
K=20

python data_utils.py $news_dir $segment_dir $vocab_file $token_ids_dir
python pmi.py $vocab_file $corpus_dir $polar_seed_file $K