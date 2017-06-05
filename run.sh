#!/bin/bash

start_step=1
end_step=1

output_dir='../output'
raw_news_dir='../data/CnNewsReport'
raw_price_dir='../data/CnStockPrice'
mkdir -p $output_dir

# process news corpus and prices
segment_dir=$output_dir/segmented
news_dir=$output_dir/token_ids
vocab_file=$output_dir/vocab
price_dir=$output_dir/prices
embedding_train_file=$output_dir/embedding_train_corpus
fastText_dir=$output_dir/fastText
if [ $start_step -le 0 ]&&[ $end_step -ge 0 ]; then
  mkdir -p $segment_dir
  mkdir -p $news_dir
  mkdir -p $price_dir
  mkdir -p $fastText_dir
  python data_utils.py $news_dir $segment_dir $vocab_file $news_dir $embedding_train_file $raw_price_dir $price_dir $fastText_dir
fi

# get P and N set
K=100
polar_seed_file=$output_dir/polar_seed
polar_optimal_file=$output_dir/polar_optimal
if [ $start_step -le 1 ]&&[ $end_step -ge 1 ]; then
  python pmi.py $vocab_file $news_dir $polar_seed_file $K $polar_optimal_file
fi

# train glove word embedding
glove_output_dir=$output_dir/glove
glove_embedding=$glove_output_dir/vectors.txt
if [ $start_step -le 2 ]&&[ $end_step -ge 2 ]; then
  sh train_glove.sh $embedding_train_file $glove_output_dir vectors
fi

# train word2vec word embedding
word2vec_output_dir=$output_dir/word2vec
word2vec_embedding=$word2vec_output_dir/vectors.txt
if [ $start_step -le 3 ]&&[ $end_step -ge 3 ]; then
  sh train_word2vec.sh $embedding_train_file $word2vec_output_dir vectors.txt
fi

# train fastText graph embedding
fastText_embedding=$fastText_dir/vectors.txt
if [ $start_step -le 4 ]&&[ $end_step -ge 4 ]; then
  train_file=$fastText_dir/train_file
  test_file=$fastText_dir/test_file
  sh train_fastText.sh $train_file $test_file $fastText_dir vectors.txt
fi

# train model
train_dir=../output/train_data
if [ $start_step -le 5 ]&&[ $end_step -ge 5 ]; then
  mkdir -p $train_dir
  python model.py --train_dir=$train_dir
fi

# test model
if [ $start_step -le 6 ]&&[ $end_step -ge 6 ]; then
  python model.py --test --train_dir=$train_dir
fi
