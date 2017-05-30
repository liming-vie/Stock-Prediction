#!/usr/bin/env python
# encoding: utf-8

__author__ = 'liming-vie'

import data_utils
import numpy as np
import tensorflow as tf

from data_utils import News
from data_utils import Price
from data_utils import StockInfo

tf.app.flags.DEFINE_string('price_dir', '../data/CnStockPrice', 'directory of price files')
tf.app.flags.DEFINE_string('news_dir', '../output/token_ids', 'directory of news files')
tf.app.flags.DEFINE_string('train_dir', '../output/train_data', 'directory for saving training files')

tf.app.flags.DEFINE_string('glove_file', '../output/glove/vectors.txt', 'glove embedding file')
tf.app.flags.DEFINE_string('word2vec_file', '../output/word2vec/vectors.txt', 'word2vec embedding file')

tf.app.flags.DEFINE_boolean('test', False, 'set to True for predict task')
tf.app.flags.DEFINE_float('test_ratio', 0.2, 'use test_ratio of data for test')

tf.app.flags.DEFINE_integer('title_max_length', 75, 'max length for news title')
tf.app.flags.DEFINE_integer('content_max_length', 32077,'max length for news content')
tf.app.flags.DEFINE_integer('period_max_length', 960, 'max length for time period')
tf.app.flags.DEFINE_integer('period_min_length', 5, 'min length for time period')

tf.app.flags.DEFINE_integer('title_units', 64, 'number of units in title birnn forward and backward cells')
tf.app.flags.DEFINE_integer('title_layers', 1, 'number of layers in title birnn forward and backward cells')
tf.app.flags.DEFINE_integer('content_units', 256, '')
tf.app.flags.DEFINE_integer('content_layers', 2, '')
tf.app.flags.DEFINE_integer('price_units', 128, '')
tf.app.flags.DEFINE_integer('price_layers', 3, '')

tf.app.flags.DEFINE_integer('fc_layers', 4, 'number of full connect layers')
tf.app.flags.DEFINE_string('fc_units', '1024, 512, 256, 512', 'number of units in full connect layers')

tf.app.flags.DEFINE_float('l2_coef', 0.1, 'L2 regularizer coeficient')
tf.app.flags.DEFINE_float('init_lr', 0.5, 'initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.95, 'learning rate decay coef')
tf.app.flags.DEFINE_integer('batch_size', 32, 'training batch size')
tf.app.flags.DEFINE_integer('train_step', 30000, 'number of training step')

FLAGS = tf.app.flags.FLAGS

class StockPrediction:
  def __init__(self):
    # load embedding
    glove = data_utils.load_glove(FLAGS.glove_file)
    word2vec = data_utils.load_word2vec(FLAGS.word2vec_file)
    merged_embed = self.merge_glove_word2vec(glove, word2vec)
    self.vocab_size = len(merged_embed)

    FLAGS.fc_units = map(int, FLAGS.fc_units.split(','))

    self.session = tf.Session()

    ''' graph '''
    with tf.variable_scope('inputs'):
      self.training = tf.placeholder(tf.bool, name='training') 

      self.title = tf.placeholder(tf.int32, [None, FLAGS.title_max_length]) # [batch size, sequence length]
      self.content = tf.placeholder(tf.int32, [None, FLAGS.content_max_length])
      self.title_length = tf.placeholder(tf.int32, [None]) # [batch size]
      self.content_length = tf.placeholder(tf.int32, [None])

      self.price = tf.placeholder(tf.float32, [None, FLAGS.period_max_length, 7])
      self.price_length = tf.placeholder(tf.int32, [None])

      self.label = tf.placeholder(tf.float32, [None])

    with tf.variable_scope('birnn_embed'):
      self.embed_matrix = tf.Variable(merged_embed, dtype=tf.float32,
        name='embedding_matrix')
      title_embed = self.embed_birnn(FLAGS.title_units, FLAGS.title_layers, 
        self.title, self.title_length, 'title_embed_birnn')
      content_embed = self.embed_birnn(FLAGS.content_units, FLAGS.content_layers,
        self.content, self.content_length, 'content_embed_birnn')
      price_embed = self.birnn(FLAGS.price_units, FLAGS.price_layers,
        self.price, self.price_length, 'price_birnn')
      final_embed = tf.concat([title_embed, content_embed, price_embed], 1)

    with tf.variable_scope('full_connect'):
      fc_inputs = final_embed
      for i in range(FLAGS.fc_layers):
        with tf.variable_scope('full_connect_layer_%d'%i):
          fc_outputs = tf.contrib.layers.legacy_full_connected(
            fc_inputs, FLAGS.fc_units[i], 
            activation=tf.nn.relu,
            weight_regularizer=tf.contrib.layers.l2_regularizer(
              FLAGS.l2_coef))
          fc_inputs = fc_outputs

    with tf.variable_scope('dropout'):
      dropout = tf.layers.dropout(fc_outputs, training=self.training)

    with tf.variable_scope('output'):
      W = tf.get_variable('W', shape=[FLAGS.fc_units[-1], 2], 
        initializer=tf.truncated_normal_initializer())
      biases = tf.get_variable('biases', shape=[2], 
        initializer=tf.random_normal_initializer())
      logits=tf.matmul(dropout, W)+biases
      self.result = tf.nn.softmax(logits)

    with tf.variable_scope('train'):
      self.cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits))

      self.learning_rate = tf.Variable(FLAGS.init_lr, trainable=False, 
        name="learning_rate")
      self.lr_decay_op = self.learning_rate.assign(
        self.learning_rate * FLAGS.lr_decay)

      self.global_step = tf.Variable(0, trainable=False, name='global_step')
      self.train_op = tf.train.AdamOptimizer(FLAGS.init_lr) \
          .minimize(self.cross_entropy, self.global_step)

    with tf.variable_scope('logs'):
      self.saver = tf.train.Saver(tf.global_variables())
      self.log_writer=tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir, 'logs/'), self.session.graph)
      self.summary = tf.Summary()


  def lstm_cell(self, num_units, num_layers, scope):
    def cell():
      return tf.contrib.rnn.LSTMCell(num_units)
    with tf.variable_scope(scope):
      if num_layers > 1:
        return tf.contrib.rnn.MultiRNNCell([cell() for _ in xrange(num_layers)])
    return cell()

  def birnn(self, units, layers, inputs, input_length, scope):
    with tf.variable_scope(scope):
      fw_cell = self.lstm_cell(units, layers, 'forward')
      bw_cell = self.lstm_cell(units, layers, 'backward')
      _, states = tf.nn.bidirectional_dynamic_rnn(
        fw_cell, bw_cell, inputs, 
        sequence_length=input_length,
        dtype=tf.float32)

      def proc_multi_layer_state(state, num_units, num_layers, scope):
        with tf.variable_scope(scope):
          # from [num_layers, batch_size, num_units] 
          # to [batch_size, num_layers, num_units]
          state = tf.transpose(state, perm=[2, 0, 1, 3])
          # [batch_size, num_layers*num_units]
          return tf.reshape(state, [-1, num_units*num_layers*2])

      def proc_single_layer_state(state, num_units, scope):
        with tf.variable_scope(scope):
          # from [2, batch_size, num_units] 
          # to [batch_size, 2, num_units]
          state = tf.transpose(state, perm=[1, 0, 2])
          # [batch_size, num_layers*num_units]
          return tf.reshape(state, [-1, num_units*2])

      if layers > 1:
        state_fw = proc_multi_layer_state(states[0], units, layers, 'forward_state')
        state_bw = proc_multi_layer_state(states[1], units, layers, 'backward_state')
      else:
        state_fw = proc_single_layer_state(states[0], units, 'forward_state')
        state_bw = proc_single_layer_state(states[1], units, 'backward_state')

      return tf.concat([state_fw, state_bw], 1)

  def embed_birnn(self, units, layers, inputs, input_length, scope):
    with tf.variable_scope(scope):
      embed=tf.nn.embedding_lookup(self.embed_matrix, inputs)
    return self.birnn(units, layers, embed, input_length, scope)

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

  def process_tokens(self, sequence):
    return map(lambda x: x if x<self.vocab_size else -1, sequence)

  def news_idx_for_date(self, news, date):
    for i, n in enumerate(news):
      if n.date > date:
        return i
    return len(news)

  def stock_info_iter(self, train):
    stock_info = data_utils.load_stock_with_news(
      FLAGS.price_dir, FLAGS.news_dir)
    prices, titles, contents=[], [], []

    def add_data(date, price, info):
      idx = self.news_idx_for_date(info.news, date)


    for code, info in enumerate(stock_info):
      size = int(len(info.prices)*self.test_ratio)
      if train:
        all_prices = info.prices[:-size]
        size = len(info.prices) - size
      else:
        all_prices = info.prices[-size:]

      cur_prices=[]
      for price in all_prices:
        cur_prices.append([i[1] for i in price._asdict().items()[1:]])
        if not train or len(cur_prices) >= FLAGS.period_min_length:
          add_data(price.date, cur_prices, info)
      if train and size > 0 and size < FLAGS.period_min_length:
        add_data(info.prices[-1].date, cur_prices, info)

    epoch = 1
    if train:
      epoch = 10000000

    data_size = len(prices)
    for _ in epoch:
      if train:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        prices = prices[shuffle_indices]
        titles = titles[shuffle_indices]
        contenst = titles[shuffle_indices]


  def make_input(self):
    pass

  def train(self):
    pass

  def test(self):
    pass


def main(_):
  if FLAGS.test:
    StockPrediction().test()
  else:
    StockPrediction().train()


if __name__ == '__main__':
  tf.app.run()
