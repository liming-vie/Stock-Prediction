#!/usr/bin/env python
# encoding: utf-8

__author__ = 'liming-vie'

import os
from tqdm import tqdm
from pyltp import Segmentor


ltp_segmentor = Segmentor()
ltp_segmentor.load('/home/olivia/ltp_data/cws.model')


def segment_corpus(input_dir, output_dir):
  print 'Segmenting news corpus...'
  for fname in tqdm(os.listdir(input_dir)):
    with open(os.path.join(output_dir, fname), 'w') as fout:
      for line in open(os.path.join(input_dir, fname)):
        ps = [i.strip() for i in line.rstrip().split('\t')]
        ps[-1] = ' '.join(ltp_segmentor.segment(ps[-1]))
        ps[-3] = ' '.join(ltp_segmentor.segment(ps[-3]))
        fout.write('\t'.join(ps)+'\n')


def get_vocab(input_dir, output_file):
  def add_to_vocab(dic, snt):
    tokens = snt.split()
    for token in tokens:
      if token in dic:
        dic[token] += 1
      else:
        dic[token] = 1

  print 'Counting tokens...'
  vocab = {}
  for fname in tqdm(os.listdir(input_dir)):
    for line in open(os.path.join(input_dir, fname)):
      ps = line.rstrip().split('\t')
      add_to_vocab(vocab, ps[-1])
      add_to_vocab(vocab, ps[-3])

  print 'Sorting vocab list...'
  sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

  print 'Writing vocab info in file %s'%output_file
  idx = 0
  with open(output_file, 'w') as fout:
    for k, v in sorted_vocab:
      fout.write("%s %d\n"%(k, v))
      vocab[k] = idx
      idx += 1
  return vocab


def transform_corpurs_to_token_ids(input_dir, output_dir, vocab):
  def transform_to_token_ids(snt, vocab):
    tokens = snt.split()
    for i, t in enumerate(tokens):
      tokens[i] = str(vocab[t])
    return ' '.join(tokens)

  print 'Transforming news corpus into token ids...'
  for fname in tqdm(os.listdir(input_dir)):
    with open(os.path.join(output_dir, fname), 'w') as fout:
      for line in open(os.path.join(input_dir, fname)):
        ps = line.rstrip().split('\t')
        ps[-3] = transform_to_token_ids(ps[-3], vocab)
        ps[-1] = transform_to_token_ids(ps[-1], vocab)
        fout.write('\t'.join(ps)+'\n')


if __name__ == '__main__':
  news_dir = '../data/CnNewsReport'
  segment_dir = '../output/segmented'
  vocab_file = '../output/vocab'
  token_ids_dir = '../output/token_ids'

  segment_corpus(news_dir, segment_dir)
  vocab = get_vocab(segment_dir, vocab_file)
  transform_corpurs_to_token_ids(segment_dir, token_ids_dir, vocab)