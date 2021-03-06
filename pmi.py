#!/usr/bin/env python
# encoding: utf-8

__author__ = 'liming-vie'

import sys
import os
import math
import data_utils
from tqdm import tqdm

P_seed = ['受益', '提升', '改善', '稳健', '看好', '有望', '增', '收购', '利好', '优势']
N_seed = ['下滑', '低于', '下降', '拖累', '跌', '降', '亏损', '违规', '处罚', '利空']


def get_seed_idx(p_seed, n_seed, vocab2idx):
  p_idx = [vocab2idx[i] for i in p_seed]
  n_idx = [vocab2idx[i] for i in n_seed]
  return p_idx, n_idx


def process_corpus(corpus):
  print 'Processing corpus...'
  for i, doc in tqdm(enumerate(corpus)):
    dic={}
    for token in doc:
      dic[token]=True
    corpus[i] = dic
  return corpus


def Pw(corpus, v):
  count = 0
  for doc in corpus:
    if v in doc:
      count += 1
  return count / float(len(corpus))


def get_Pw(corpus, vocab_size):
  print 'Calculating P(w)...'
  ret = [0. for _ in xrange(vocab_size)]
  for v in tqdm(xrange(vocab_size)):
    ret[v] = Pw(corpus, v)
  return ret


def get_set_Pw(corpus, p_idx, n_idx):
  ppw = [Pw(corpus, v) for v in p_idx]
  npw = [Pw(corpus, v) for v in n_idx]
  return ppw, npw


def Pwv(corpus, w, v):
  count = 0
  for doc in corpus:
    if w in doc and v in doc:
      count+=1
  return float(count) / len(corpus)


def pmi(corpus, pw, pv, w, v):
  pwv = Pwv(corpus, w, v)
  if pwv == 0.0:
    return 0.
  return math.log(pwv / (pw * pv));


def polar(corpus, p_idx, n_idx, ppw, npw, wi, pw):
  pp = 0.
  for i, pi in enumerate(p_idx):
    pp += pmi(corpus, pw, ppw[i], wi, pi)
  pn = 0.
  for i, ni in enumerate(n_idx):
    pn += pmi(corpus, pw, npw[i], wi, ni)
  return pp/len(p_idx) - pn/len(n_idx)


def get_polar_seed(corpus, vocab_size, vocab2idx):
  p_w = get_Pw(corpus, vocab_size)

  print 'Calculating polar_seed...'
  polar_seed = [0. for _ in xrange(vocab_size)]
  p_idx, n_idx = get_seed_idx(P_seed, N_seed, vocab2idx)
  ppw, npw = get_set_Pw(corpus, p_idx, n_idx)

  for wi in tqdm(xrange(len(p_w))):
    polar_seed[wi] = polar(corpus, p_idx, n_idx, ppw, npw, wi, p_w[wi])

  return sorted(enumerate(polar_seed), key=lambda x: x[1], reverse=True)


def get_optimal_set(K, vocab_file, corpus_dir, polar_seed_file):
  def get_set(polar_seed, K):
    return polar_seed[:K], polar_seed[-K:]

  if os.path.exists(polar_seed_file):
    polar_seed = []
    for line in open(polar_seed_file):
      ps = line.split('\t')
      polar_seed.append(int(ps[1]))
    return get_set(polar_seed, K)

  vocab2idx, vocab_str, vocab_count = data_utils.load_vocab(vocab_file)
  vocab_size = len(vocab_count)
  for i, c in enumerate(vocab_count):
    if c < 5000:
      vocab_size = i
      break
  vocab_str = vocab_str[:vocab_size]

  corpus = data_utils.load_news_corpus(corpus_dir)
  corpus = process_corpus(corpus)

  polar_seed = get_polar_seed(corpus, vocab_size, vocab2idx)

  print 'Saving polar seed in file %s'%polar_seed_file
  with open(polar_seed_file, 'w') as fout:
    for i, polar in polar_seed:
      fout.write("%s\t%d\t%f\n"%(vocab_str[i], i, polar))

  return get_set(polar_seed, K)


def save_polar_optimal(K, vocab_file, corpus_dir, polar_seed_file, polar_optimal_file):
  optimal_p, optimal_n = get_optimal_set(K, vocab_file, corpus_dir, polar_seed_file)

  corpus = data_utils.load_news_corpus(corpus_dir)
  corpus = process_corpus(corpus)

  vocab2idx, vocab_str, vocab_count = data_utils.load_vocab(vocab_file)
  vocab_size = len(vocab_count)
  for i, c in enumerate(vocab_count):
    if c < 500:
      vocab_size = i
      break
  vocab_str = vocab_str[:vocab_size]

  p_w = get_Pw(corpus, vocab_size)
  print 'Calculating porlar_optimal...'
  porlar_optimal = [0. for _ in xrange(vocab_size)]
  ppw, npw = get_set_Pw(corpus, optimal_p, optimal_n)

  for wi in tqdm(xrange(len(p_w))):
    porlar_optimal[wi] = polar(corpus, optimal_p, optimal_n, ppw, npw, wi, p_w[wi])

  porlar_optimal = sorted(enumerate(porlar_optimal), key=lambda x: x[1], reverse=True)

  print 'Saving polar optimal in file %s'%polar_optimal_file
  with open(polar_optimal_file, 'w') as fout:
    for i, p in porlar_optimal:
      fout.write("%s\t%d\t%f\n"%(vocab_str[i], i, p))


if __name__=='__main__':
  if len(sys.argv) != 6:
    print 'Usage: python pmi.py vocab_file corpus_dir polar_seed_file K polar_optimal_file'
    sys.exit(1)

  vocab_file, corpus_dir, polar_seed_file, K, polar_optimal_file=sys.argv[1:]
  K = int(K)

  save_polar_optimal(K, vocab_file, corpus_dir, polar_seed_file, polar_optimal_file)