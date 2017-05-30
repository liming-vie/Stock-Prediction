#!/bin/bash
output_dir='../output'
raw_news_dir='../data/CnNewsReport'
price_dir='../data/CnStockPrice'
mkdir -p $output_dir

# process news corpus
segment_dir=$output_dir/segmented
news_dir=$output_dir/token_ids
vocab_file=$output_dir/vocab
embedding_train_file=$output_dir/embedding_train_corpus
mkdir -p $segment_dir
mkdir -p $news_dir
python data_utils.py $news_dir $segment_dir $vocab_file $news_dir $embedding_train_file

# get P and N set
polar_seed_file=$output_dir/polar_seed
K=100
python pmi.py $vocab_file $news_dir $polar_seed_file $K

# train glove word embedding
glove_output_dir=$output_dir/glove
glove_embedding=$glove_output_dir/vectors.txt
mkdir -p $glove_output_dir
sh train_glove.sh $embedding_train_file $glove_output_dir vectors

# train word2vec word embedding
word2vec_output_dir=$output_dir/word2vec
word2vec_embedding=$word2vec_output_dir/vectors.txt
mkdir -p $word2vec_output_dir
sh train_word2vec.sh $embedding_train_file $word2vec_embedding

# train fastText graph embedding
fastText_output_dir=$output_dir/fastText
mkdir -p $fastText_output_dir
sh train_fastText.sh $price_dir $news_dir embed_train_file $fastText_output_dir model

train_dir=../output/train_data
mkdir -p $train_dir
python model.py
python model.py --test