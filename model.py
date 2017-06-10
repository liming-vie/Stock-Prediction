#!/usr/bin/env python
# encoding: utf-8

__author__ = 'liming-vie'

import os

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

import sys
import random
import data_utils
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from data_utils import News
from data_utils import Price
from data_utils import StockData

tf.app.flags.DEFINE_string('price_dir', '../output/prices', 'directory of price files')
tf.app.flags.DEFINE_string('news_dir', '../output/token_ids', 'directory of news files')
tf.app.flags.DEFINE_string('train_dir', '../output/train_data', 'directory for saving training files')
tf.app.flags.DEFINE_string('test_output', '../output/test_result', 'file of test result')

tf.app.flags.DEFINE_string('glove_file', '../output/glove/vectors.txt', 'glove embedding file')
tf.app.flags.DEFINE_string('word2vec_file', '../output/word2vec/vectors.txt', 'word2vec embedding file')
tf.app.flags.DEFINE_string('fastText_doc_file', '../output/fastText/test_file', 'fastText doc file')
tf.app.flags.DEFINE_string('fastText_vector_file', '../output/fastText/vectors.txt', 'fastText embedding file')

tf.app.flags.DEFINE_boolean('test', False, 'set to True for predict task')
tf.app.flags.DEFINE_float('test_ratio', 0.2, 'use test_ratio of data for test')
tf.app.flags.DEFINE_integer('batch_size', 32, '')

tf.app.flags.DEFINE_integer('content_max_length', 5000,'max length for news content')
tf.app.flags.DEFINE_integer('title_max_length', 75,'max length for news content')
tf.app.flags.DEFINE_integer('period_max_length', 30, 'min length for time period')
tf.app.flags.DEFINE_integer('period_min_length', 1, 'min length for time period')

tf.app.flags.DEFINE_integer('title_units', 64, 'number of units in title birnn forward and backward cells')
tf.app.flags.DEFINE_integer('title_layers', 2, 'number of layers in title birnn forward and backward cells')
tf.app.flags.DEFINE_integer('content_units', 128, '')
tf.app.flags.DEFINE_integer('content_layers', 3, '')
tf.app.flags.DEFINE_integer('price_units', 64, '')
tf.app.flags.DEFINE_integer('price_layers', 3, '')
tf.app.flags.DEFINE_integer('doc_units', 128, '')
tf.app.flags.DEFINE_integer('doc_layers', 4, '')

tf.app.flags.DEFINE_integer('fc_layers', 4, 'number of full connect layers')
tf.app.flags.DEFINE_string('fc_units', '512, 512, 256, 256', 'number of units in full connect layers')

tf.app.flags.DEFINE_float('l2_coef', 0.1, 'L2 regularizer coeficient')
tf.app.flags.DEFINE_float('init_lr', 0.001, 'initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.95, 'learning rate decay coef')
tf.app.flags.DEFINE_integer('train_steps', 30000, 'number of training step')
tf.app.flags.DEFINE_integer('ckpt_per_steps', 100, 'save checkpoint per ckpt_per_steps steps')

FLAGS = tf.app.flags.FLAGS

class StockPrediction:
  def __init__(self):
    # load word embedding
    glove = data_utils.load_glove(FLAGS.glove_file)
    word2vec = data_utils.load_word2vec(FLAGS.word2vec_file)
    merged_embed, self.vocab_size = self.merge_glove_word2vec(glove, word2vec)
    dim = len(merged_embed[0])
    merged_embed.append([0. for _ in xrange(dim)])

    # load doc embedding
    self.doc_embedding, doc_dim = data_utils.load_fastText_embed(\
      FLAGS.fastText_doc_file, FLAGS.fastText_vector_file)
    self.zero_doc_key = self.doc_key([self.vocab_size], [self.vocab_size])
    self.doc_embedding[self.zero_doc_key] = [0. for _ in xrange(doc_dim)]


    FLAGS.fc_units = map(int, FLAGS.fc_units.split(','))

    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.session = tf.Session(config=config)

    ''' graph '''
    print 'Initializing model graph...'
    with tf.variable_scope('inputs'):
      self.training = tf.placeholder(tf.bool, name='training')

      self.title = tf.placeholder(tf.int32, shape=[None, None],
        name='title') # [batch size, sequence length]
      self.content = tf.placeholder(tf.int32, shape=[None, None],
        name='content')
      self.title_length = tf.placeholder(tf.int32, shape=[None], \
        name='title_length')
      self.content_length = tf.placeholder(tf.int32, shape=[None],\
        name='content_length')

      self.prices = tf.placeholder(tf.float32, name='prices', \
        shape=[None, None, 7])
      self.price_length = tf.placeholder(tf.int32, shape=[None], \
        name='price_length')

      self.docs = tf.placeholder(tf.float32, name='docs', \
        shape=[None, None, doc_dim])
      self.doc_length = tf.placeholder(tf.int32, shape=[None], \
        name='doc_length')

      self.label = tf.placeholder(tf.int32, shape=[None, 2], name='label')

    with tf.variable_scope('birnn_embed'):
      self.word_embedding = tf.Variable(merged_embed, dtype=tf.float32,
        name='word_embedding_matrix')
      title_embed = self.embed_birnn(FLAGS.title_units, FLAGS.title_layers,
        self.title, self.title_length, scope='title_embed_birnn')
      content_embed = self.embed_birnn(FLAGS.content_units, FLAGS.content_layers,
        self.content, self.content_length, scope='content_embed_birnn')
      price_embed = self.birnn(FLAGS.price_units, FLAGS.price_layers,
        self.prices, self.price_length, scope='price_birnn')
      doc_embed = self.birnn(FLAGS.doc_units, FLAGS.doc_layers,
        self.docs, self.doc_length, scope='doc_birnn')
      final_embed = tf.concat([title_embed, content_embed, doc_embed, price_embed], 1)

    with tf.variable_scope('full_connect'):
      fc_inputs = final_embed
      for i in range(FLAGS.fc_layers):
        with tf.variable_scope('full_connect_layer_%d'%i):
          fc_outputs = tf.contrib.layers.legacy_fully_connected(
            fc_inputs, FLAGS.fc_units[i],
            activation_fn=tf.nn.relu,
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
      self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
      self.log_writer=tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir, 'logs/'), self.session.graph)
      self.summary = tf.Summary()


  def lstm_cell(self, num_units, num_layers, scope='lstm_cell'):
    def cell():
      return tf.contrib.rnn.LSTMCell(num_units)
    with tf.variable_scope(scope):
      if num_layers > 1:
        return tf.contrib.rnn.MultiRNNCell([cell() for _ in xrange(num_layers)])
    return cell()

  def birnn(self, units, layers, inputs, input_length, scope='birnn'):
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
          # to [batch_size, 2, num_layers, num_units]
          state = tf.transpose(state, perm=[2, 0, 1, 3])
          # [batch_size, num_layers*num_units]
          return tf.reshape(state, [-1, num_units*num_layers*2])

      def proc_single_layer_state(state, num_units, scope):
        with tf.variable_scope(scope):
          # from [2, batch_size, num_units]
          # to [batch_size, 2, num_units]
          state = tf.transpose(state, perm=[1, 0, 2])
          return tf.reshape(state, [-1, num_units*2])

      if layers > 1:
        state_fw = proc_multi_layer_state(states[0], units, layers, 'forward_state')
        state_bw = proc_multi_layer_state(states[1], units, layers, 'backward_state')
      else:
        state_fw = proc_single_layer_state(states[0], units, 'forward_state')
        state_bw = proc_single_layer_state(states[1], units, 'backward_state')

      return tf.concat([state_fw, state_bw], 1)

  def embed_birnn(self, units, layers, inputs, input_length, scope='embed_birnn'):
    with tf.variable_scope(scope):
      embed=tf.nn.embedding_lookup(self.word_embedding, inputs)
    return self.birnn(units, layers, embed, input_length, scope)


  def doc_key(self, title, content):
    title_str = ' '.join(map(str, title))
    content_str = ' '.join(map(str, content))
    return hash("%s %s"%(title_str, content_str))

  def get_doc_embed_with(self, title, content):
    return self.doc_embedding.get(self.doc_key(title, content),\
      self.doc_embedding[self.zero_doc_key])


  def merge_glove_word2vec(self, glove, word2vec):
    l=len(glove) # assume glove and word2vec has the same vocab
    ret = [0 for _ in xrange(l)]
    for i in xrange(l):
      ret[i] = glove[i] + word2vec[i]
    return ret, l


  def make_input(self, batch_data, training=True):
    zero_id=self.vocab_size

    def get_titles_and_contents(info):
      if len(info.news) == 0:
        return [[[zero_id]], [1], [[zero_id]], [1]]

      titles, contents = [], []
      title_length, content_length = [], []
      for news in info.news:
        if not news:
          titles.append([zero_id])
          contents.append([zero_id])
        else:
          idx=random.randint(0, len(news)-1)
          titles.append(news[idx].title)
          contents.append(news[idx].content)
        title_length.append(len(titles[-1]))
        content_length.append(len(contents[-1]))
      return [titles, title_length, contents, content_length]

    zero_price=[0. for _ in xrange(7)]
    def get_prices(info):
      prices=[price for price in info.prices if price]
      if len(prices)>FLAGS.period_max_length:
        prices=prices[-FLAGS.period_max_length:]
      elif len(prices) == 0:
        prices = [zero_price]
      return [prices, len(prices)]

    def align_batch_data(batch, length, max_length, padding):
      max_l = min(max_length, max(length))
      for i, data in enumerate(batch):
        if length[i] < max_l:
          batch[i].extend([padding for _ in xrange(max_l-length[i])])
        elif length[i] > max_l:
          length[i] = max_l
          batch[i] = data[:max_l]
      return [batch, length]

    def get_doc_embedding(titles, contents):
      ret = [self.get_doc_embed_with(title, content) for title, content \
          in zip(titles[-FLAGS.period_max_length:], \
              contents[-FLAGS.period_max_length:])]
      return [ret, len(ret)]

    def get_label(change):
      return [1, 0] if change<=0. else [0, 1]

    news_info = [get_titles_and_contents(info) for info in batch_data]

    docs_info = [get_doc_embedding(info[0], info[2]) for info in news_info]
    doc_embed, doc_length = align_batch_data([info[0] for info in docs_info], \
      [info[1] for info in docs_info], FLAGS.period_max_length, \
      self.doc_embedding[self.zero_doc_key])

    contents, content_length = align_batch_data([info[2][-1] for info in news_info], \
      [info[3][-1] for info in news_info], \
      FLAGS.content_max_length, zero_id)
    titles, title_length = align_batch_data([info[0][-1] for info in news_info], \
      [info[1][-1] for info in news_info], \
      FLAGS.title_max_length, zero_id)

    prices_info = [get_prices(info) for info in batch_data]
    prices, price_length = align_batch_data([info[0] for info in prices_info], \
      [info[1] for info in prices_info], \
      FLAGS.period_max_length, zero_price)

    return {
      self.training : training,
      self.label : [get_label(info.change) for info in batch_data],

      self.title : titles,
      self.title_length : title_length,

      self.content : contents,
      self.content_length : content_length,

      self.prices : prices,
      self.price_length : price_length,

      self.docs : doc_embed,
      self.doc_length : doc_length
    }


  def get_data_iter(self, train):
    stock_infos = data_utils.load_stock_with_news(\
      FLAGS.price_dir, FLAGS.news_dir, self.vocab_size)
    return data_utils.stock_info_iter(stock_infos, train, FLAGS.batch_size,\
      FLAGS.test_ratio, FLAGS.period_min_length)

  def init_model(self):
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      print ('Restoring model from %s'%ckpt.model_checkpoint_path)
      self.saver.restore(self.session, ckpt.model_checkpoint_path)
    else:
      print ('Initializing model variables')
      self.session.run(tf.global_variables_initializer())

  def train(self):
    data_iter = self.get_data_iter(True)

    with self.session.as_default():
      self.init_model()
      checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
      print 'Checkpoint directory in %s'%FLAGS.train_dir

      print 'Start training...'
      cross_entropy = 0.0
      prev_cross_entropy = [float('inf')]
      step = self.global_step.eval()
      output_feed = [self.global_step, self.train_op, self.cross_entropy]
      while step <= FLAGS.train_steps:
        step, _, ce = self.session.run(output_feed, \
          self.make_input(data_iter.next()))
        cross_entropy += ce
        # save checkpoint
        if step % FLAGS.ckpt_per_steps == 0:
          cross_entropy /= FLAGS.ckpt_per_steps
          print ("global_step %d, cross entropy %f, learning rate %f"%(
            step, cross_entropy, self.learning_rate.eval()))
          sys.stdout.flush()

          if cross_entropy > max(prev_cross_entropy):
            self.session.run(self.lr_decay_op)
          prev_cross_entropy = (prev_cross_entropy+[cross_entropy])[-5:]
          cross_entropy = 0.

          self.saver.save(self.session, checkpoint_path, \
            global_step=self.global_step)
          self.log_writer.add_summary(self.summary, step)


  def test(self):
    data_iter = self.get_data_iter(False)

    total=0
    correct=0
    with self.session.as_default(), open(FLAGS.test_output, 'w') as fout:
      fout.write('code name date label y probs\n')
      self.init_model()

      print 'Start testing...'
      for batch in tqdm(data_iter):
        input_feed=self.make_input(batch)
        results = self.session.run(self.result, input_feed)
        for info, result, label in zip(batch, results, input_feed[self.label]):
          label = np.argmax(label)
          y=np.argmax(result)
          if y==label:
            correct+=1
          total += 1
          fout.write('%s %s %s %f %d %s\n'%(   \
            info.code, info.name, info.date, info.change, y, \
              ','.join(map(str, result))))
        fout.flush()
    print 'Test done, accuracy: %f'%(float(correct)/total)
    print 'Test result saved in %s'%FLAGS.test_output


def main(_):
  if FLAGS.test:
    StockPrediction().test()
  else:
    StockPrediction().train()


if __name__ == '__main__':
  tf.app.run()
