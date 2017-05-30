#!/usr/bin/env python
# encoding: utf-8

import data_utils
import tensorflow as tf


tf.app.flags.DEFINE_boolean('infer', False, 'set to True for predict task.')

tf.app.flags.DEFINE_string('price_dir', '', 'directory of price files')
tf.app.flags.DEFINE_string('news_dir', '', 'directory of news files')

tf.app.flags.DEFINE_string('glove_file', '', 'glove embedding file')
tf.app.flags.DEFINE_string('word2vec_file', '', 'word2vec embedding file')

tf.app.flags.DEFINE_string('vocab_file', '', 'vocab file')
tf.app.flags.DEFINE_integer('min_frequency', 5, 'min frequency when loading vocab file')

FLAGS = tf.app.flags.FLAGS

class StockPrediction:
  def __init__(self):
    # load stock info
    self.stock_info = data_utils.load_stock_with_news(FLAGS.price_dir, FLAGS.news_dir)

    # load embedding
    glove = data_utils.load_glove(FLAGS.glove_file)
    word2vec = data_utils.load_word2vec(FLAGS.word2vec_file)
    merged_embed = self.merge_glove_word2vec(glove, word2vec)
    self.vocab_size = len(merged_embed)

  

  def merge_glove_word2vec(self, glove, word2vec):
    gl, wl = len(glove), len(word2vec)
    ret = [0 for _ in xrange(max(gl, wl))]
    for i in xrange(min(gl, wl)):
      ret[i] = glove[i] + word2vec[i]
    if gl>wl:
      a, b = glove, word2vec
    else:
      a, b = word2vec, glove
    for i in xrange(min(gl, wl), max(gl, wl)):
      ret[i] = a[i] + b[-1]
    return ret

  def vector(embed, idx):
    return embed[idx] if idx < self.vocab_size else embed[-1]


  def train():
    pass

  def infer():
    pass


def main(_):
  if FLAGS.infer:
    StockPrediction().train()
  else:
    StockPrediction().infer()


if __name__ == '__main__':
  tf.app.run()