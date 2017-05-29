#!/usr/bin/env python
# encoding: utf-8

__author__ = 'liming-vie'

import sys
import os

P_seed = [受益, 提升, 改善, 稳健, 看好, 有望, 增, 收购, 利好, 优势]
N_seed = [下滑, 低于, 下降, 拖累, 跌, 降, 亏损, 违规, 处罚, 利空]

def load_news_corpus(path):
  docs = []
  for fname in os.listdir(path):
    for line in open(os.path.join(path, fname)):
      ps = line.rstrip().split('\t')
      docs.append([ps[-3].strip(), ps[-1]].strip()) # title, content
  return docs

if __name__=='__main__':
  pass
