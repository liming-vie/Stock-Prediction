#!/usr/bin/env python
# encoding: utf-8

__author__ = 'liming-vie'

import os
import sys
import datetime
from tqdm import tqdm
from pyltp import Segmentor
from collections import namedtuple


News = namedtuple('News', 'date titles contents')
Price = namedtuple('Price', 'date close open change highest lowest amount turnover')
StockInfo = namedtuple('StockInfo', 'name code news prices')

def date(tstr):
  ps = tstr.split('-')
  return datetime.date(int(ps[0]), int(ps[1]), int(ps[2]))

def load_stock_with_news(price_dir, news_dir):
  basedate = datetime.date(2000, 1, 1)
  stock_infos = {}

  print 'Loading stock price info...'
  for stock_name in tqdm(os.listdir(price_dir)):
    ps = stock_name.split('.')
    if ps[1] == 'txt':
      continue
    code, name = ps
    prices = []
    for line in open(os.path.join(price_dir, stock_name)):
      ps = line.split('\t')
      tmp = map(float, ps[1:])
      tmp.insert(0, (date(ps[0]) - basedate).days)
      prices.append(Price._make(tmp))
    stock_infos[code] = StockInfo(name, code, [], prices)

  print 'Loading stock news info...'
  for fname in tqdm(os.listdir(news_dir)):
    d = (date(fname.split('_')[1]) - basedate).days
    today_news = {}
    for line in open(os.path.join(news_dir, fname)):
      ps = line.rstrip().split('\t')
      codes = ps[-2].split(',')
      for code in codes:
        if code not in today_news:
          today_news[code] = []
        today_news[code].append([
          map(int, ps[-3].split(' ')), 
          map(int, ps[-1].split(' '))])
    for code in today_news:
      stock_infos[code].news.append(
        News(date=d, 
          titles=[t[0] for t in today_news[code]], 
          contents=[t[1] for t in today_news[code]]))
  for code in stock_infos:
    stock_infos[code]._replace(
      news=sorted(stock_infos[code].news, key=lambda x: x.date))

  return stock_infos

def load_word2vec(embed_file):
  print 'Loading word2vec embedding...'
  with open(embed_file) as fin:
    size, dim = map(int, fin.readline().split())
    ret = [0. for _ in xrange(size)]
    for line in tqdm(fin.readlines()):
      ps = line.rstrip().split()
      vec = map(float, ps[1:])
      try:
        ret[int(ps[0])] = vec
      except:
        ret[-1] = vec
  return ret


def load_glove(embed_file):
  print 'Loading glove embedding...'
  lines = open(embed_file).readlines()
  ret = [0. for _ in xrange(len(lines))]
  for line in tqdm(lines):
    ps = line.rstrip().split()
    vec = map(float, ps[1:])
    try:
      ret[int(ps[0])] = vec
    except:
      ret[-1] = vec
  return ret


def segment_corpus(input_dir, output_dir):
  print 'Segmenting news corpus...'
  ltp_segmentor = Segmentor()
  ltp_segmentor.load('/home/olivia/ltp_data/cws.model')
  for fname in tqdm(os.listdir(input_dir)):
    with open(os.path.join(output_dir, fname), 'w') as fout:
      for line in open(os.path.join(input_dir, fname)):
        ps = [i.strip() for i in line.rstrip().split('\t')]
        ps[-1] = ' '.join(ltp_segmentor.segment(ps[-1]))
        ps[-3] = ' '.join(ltp_segmentor.segment(ps[-3]))
        fout.write('\t'.join(ps)+'\n')


def get_vocab2idx(input_dir, output_file):
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
      fout.write("%s\t%d\n"%(k, v))
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


def load_news_corpus(path):
  print 'Loading news corpus...'
  docs = []
  for fname in tqdm(os.listdir(path)):
    for line in open(os.path.join(path, fname)):
      ps = line.rstrip().split('\t')
      text = "%s %s"%(ps[-3], ps[-1]) # title, content
      docs.append(map(int, text.split()))
  return docs

def load_vocab(file_path):
  print 'Loading vocab file...'
  vocab_count = []
  vocab2idx = {}
  vocab_str = []
  idx=0
  for line in tqdm(open(file_path)):
    ps=line.split('\t')
    vocab2idx[ps[0]] = idx
    vocab_count.append(int(ps[1]))
    vocab_str.append(ps[0])
    idx+=1
  return vocab2idx, vocab_str, vocab_count


def saving_corpus(news_dir, output_file):
  corpus = load_news_corpus(news_dir)
  print 'Saving corpus in file %s'%output_file
  with open(output_file, 'w') as fout:
    fout.write(' '.join([str(i) for i in doc])+'\n')


if __name__ == '__main__':
  if len(sys.argv) != 6:
    print 'Usage: python data_utils.py news_dir segment_dir vocab_file token_ids_dir embedding_train_corpus'
    sys.exit(1)

  news_dir, segment_dir, vocab_file, token_ids_dir, embedding_train_file = sys.argv[1:]
  
  segment_corpus(news_dir, segment_dir)
  vocab2idx = get_vocab2idx(segment_dir, vocab_file)
  transform_corpurs_to_token_ids(segment_dir, token_ids_dir, vocab2idx)
  saving_corpus(token_ids_dir, embedding_train_file)
