#!/usr/bin/env python
# encoding: utf-8

__author__ = 'liming-vie'

import os
import sys
import copy
import datetime
import numpy as np
from tqdm import tqdm
from pyltp import Segmentor
from collections import namedtuple

News = namedtuple('News', 'title content')
Price = namedtuple('Price', 'close open change highest lowest amount turnover')
StockInfo = namedtuple('StockInfo', 'name code news prices dates trading_days')
StockData = namedtuple('StockData', 'name code news prices change date')


def str2date(tstr):
  ps = tstr.split('-')
  return datetime.date(int(ps[0]), int(ps[1]), int(ps[2]))


def load_stock_with_news(price_dir, news_dir, vocab_size):
  basedate = datetime.date(2000, 1, 1)
  curdate = (datetime.date.today() - basedate).days
  stock_infos = {} # code2info
  days2info = {} 

  def diff_days(d1, d2):
    return (d1-d2).days

  def days2date(days):
    return str(basedate+datetime.timedelta(days))

  print 'Loading stock price info...'
  for stock_name in tqdm(os.listdir(price_dir)):
    ps = stock_name.split('.')
    code, name = ps
    days2info[code] = {}
    for line in open(os.path.join(price_dir, stock_name)):
      ps = line.split('\t')
      days = diff_days(str2date(ps[0]), basedate)
      days2info[code][days] = [Price._make(map(float, ps[1:])), None]
    stock_infos[code] = StockInfo(name, code, None, None, None, \
      trading_days=len(days2info[code]))

  unk_id = vocab_size-1
  def process_sequence(sequence):
    tokens = map(int, sequence.split(' '))
    return map(lambda x: x if x<vocab_size else unk_id, tokens)

  print 'Loading stock news info...'
  for fname in tqdm(os.listdir(news_dir)):
    d = diff_days(str2date(fname.split('_')[1]), basedate)
    for line in open(os.path.join(news_dir, fname)):
      ps = line.rstrip().split('\t')
      codes = ps[-2].split(',')
      for code in codes:
        if d not in days2info[code]:
          days2info[code][d] = [None, []]
        elif days2info[code][d][1] == None:
          days2info[code][d][1] = []
        days2info[code][d][1].append(News._make([
          process_sequence(ps[-3]), 
          process_sequence(ps[-1])]))

  print 'Sorting news and prices by date...'
  for code in tqdm(days2info):
    infos = sorted(days2info[code].items(), key=lambda info: info[0])
    stock_infos[code] = stock_infos[code]._replace( \
      dates=[days2date(info[0]) for info in infos], \
      prices=[info[1][0] for info in infos], \
      news=[info[1][1] for info in infos])
    '''
    print code, stock_infos[code].trading_days
    print stock_infos[code].dates
    print stock_infos[code].prices
    print stock_infos[code].news
    '''

  return stock_infos


def stock_info_iter(stock_infos, train, batch_size, test_ratio, period_min_length):
  stock_infos=stock_infos.values()
  indices=np.arange(len(stock_infos))

  idx_range=[]
  for info in stock_infos:
    size = int(info.trading_days*test_ratio)
    idx,count=len(info.prices)-1, 0
    while count < size:
      if info.prices[idx] != None:
        count+=1
      idx-=1
    idx+=1
    idx_range.append(idx)

  def make_data(code, name, prices, news, date):
    return StockData(name, code, news[:-1], prices[:-1],
        change=prices[-1].change, date=date)

  infos = []
  for _ in xrange(100000000 if train else 1):
    if train:
      shuffle_indices = np.random.permutation(indices)
      stock_infos=[stock_infos[i] for i in shuffle_indices]
      idx_range=[idx_range[i] for i in shuffle_indices]

    for vi, info in zip(idx_range, stock_infos):
      if len(info.prices) == 0:
        continue

      prices, news, latest_date = [], [], None
      if not train:
        for i in xrange(0, vi, 1):
          news.append(info.news[i])
          if info.prices[i] != None:
            prices.append(info.prices[i])

      for i in xrange(vi, len(info.news), 1):
        news.append(info.news[i])
        if info.prices[i] != None:
          latest_date = info.dates[i]
          prices.append(info.prices[i])
          if not train or len(prices) > period_min_length:
            infos.append(make_data(info.code, info.name, prices, news, latest_date))
            if len(infos)  == batch_size:
              yield infos
              infos=[]        

      if train and len(prices) <= period_min_length:
        infos.append(make_data(info.code, info.name, prices, news, latest_date))
        if len(infos)  == batch_size:
          yield infos
          infos=[]
          
    if len(infos) > 0:
      yield infos


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


def price_normalization(price_dir, output_dir):
  print 'Nomalizing stock prices...'
  for fname in tqdm(os.listdir(price_dir)):
    if fname.split('.')[1] == 'txt':
      continue
    lines = map(lambda l: l.split('\t'), \
      open(os.path.join(price_dir, fname)).readlines())
    with open(os.path.join(output_dir, fname), 'w') as fout:
      if len(lines) == 0:
        continue
      dates = map(lambda l: l[0], lines)
      prices = map(lambda l: l[1:], lines)
      prices = [map(float, l) for l in prices]
      maxp = [max(abs(l[i]) for l in prices) for i in range(7)]
      for d, p in zip(dates, prices):
        fout.write('%s\t'%d)
        n_p = [p[i] / maxp[i] for i in xrange(7)]
        fout.write('\t'.join(map(str, n_p))+'\n')


def save_fastText_corpus(price_dir, news_dir, output_dir, vocab_size=132718):
  unk_id=vocab_size-1
  def process_tokens(tokens):
    return map(lambda x: x if x<vocab_size else unk_id, tokens)

  stock_infos = load_stock_with_news(price_dir, news_dir, float('inf'))

  print 'Saving news corpus in fastText training format'
  with open(os.path.join(output_dir, 'train_file'), 'w') as ftrain, \
    open(os.path.join(output_dir, 'test_file',), 'w') as ftest:
    for code, info in tqdm(stock_infos.iteritems()):
      pl = len(info.prices)
      flag = -1
      for i in xrange(pl-1, -1, -1):
        if info.news[i] != None and flag >= 0:
          for news in info.news[i]:
            ftrain.write('__label__%d %s %s\n'%(flag, \
              ' '.join(map(str, news.title)), \
              ' '.join(map(str, news.content))))
            ftest.write('%s %s\n'%( \
              ' '.join(map(str, process_tokens(news.title))),
              ' '.join(map(str, process_tokens(news.content)))))

        if info.prices[i] != None:
          flag = 1 if (info.prices[i].change > 0) else 0


def load_fastText_embed(doc_file, embed_file):
  print 'Loading fastText doc embedding...'
  ret={}
  for doc, embed in tqdm(zip(open(doc_file), open(embed_file))):
    ret[hash(doc.strip())] = map(float, embed.rstrip().split(' '))
  dim = len(ret.iteritems().next()[1])
  return ret, dim


if __name__ == '__main__':
  if len(sys.argv) != 9:
    print 'Usage: python data_utils.py news_dir segment_dir vocab_file token_ids_dir embedding_train_corpus price_dir normalized_price_dir fastText_dir'
    sys.exit(1)

  news_dir, segment_dir, vocab_file, token_ids_dir, embedding_train_file, price_dir, normalized_price_dir, fastText_dir = sys.argv[1:]
  '''
  segment_corpus(news_dir, segment_dir)
  vocab2idx = get_vocab2idx(segment_dir, vocab_file)
  transform_corpurs_to_token_ids(segment_dir, token_ids_dir, vocab2idx)
  saving_corpus(token_ids_dir, embedding_train_file)
  price_normalization(price_dir, normalized_price_dir)
  '''
  save_fastText_corpus(normalized_price_dir, token_ids_dir, fastText_dir)