# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class TransitionOperator(object):
  def __init__(self,
               alpha):
    self._prepare_network(alpha)

  def _sample(self, mu, log_sigma_sq):
    eps_shape = tf.shape(mu)
    eps = tf.random_normal( eps_shape, 0.0, 1.0, dtype=tf.float32 )
    out = tf.add(mu,
                 tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), eps))
    return out

  def _calc_log_likelihood(self, x, mu, log_sigma_sq):
    log_p = -0.5 * tf.log(2.0 * np.pi) \
            - 0.5 * log_sigma_sq \
            - tf.square(x - mu) / (2.0 * tf.exp(log_sigma_sq))
    return tf.reduce_sum(log_p, 1)

  def _prepare_network(self, alpha):
    with tf.variable_scope("transition_op"):
      self.temperature = tf.placeholder(tf.float32, shape=[],        name="temperature")
      self.x           = tf.placeholder(tf.float32, shape=[None, 2], name="x")
      
      h1 = tf.layers.dense(self.x, 4096, name="fc1", activation=tf.nn.relu)
      h2 = tf.layers.dense(h1,     4096, name="fc2", activation=tf.nn.relu)
      
      mu_org           = tf.layers.dense(h2, 2, name="mu")
      log_sigma_sq_org = tf.layers.dense(h2, 2, name="log_sigma_sq")
      
      mu = alpha * self.x + (1.0 - alpha) * mu_org
      log_sigma_sq = log_sigma_sq_org + tf.log(self.temperature)

      x_hat = self._sample(mu, log_sigma_sq)

      self.x_hat = x_hat
      self.log_p = self._calc_log_likelihood(self.x, mu, log_sigma_sq)
      
      self.mu = mu
      self.log_sigma_sq = log_sigma_sq



class VariationalWalkback(object):
  def __init__(self,
               alpha=0.5, # 大きいほど元のxに近づける
               learning_rate=1e-4,
               step_size=30,
               extra_step_size=10):
    
    self.step_size       = step_size
    self.extra_step_size = extra_step_size
    self.temperature_factor = 1.1
    
    self._prepare(alpha, learning_rate)


  def _prepare(self, alpha, learning_rate):
    self.trans_op = TransitionOperator(alpha)

    with tf.variable_scope("loss"):
      self.loss = tf.reduce_mean(-self.trans_op.log_p)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      self.train_op = optimizer.minimize(self.loss)


  def train(self, sess, data):
    xs = data
    total_loss = 0

    for i in range(self.step_size):
      print("pass loop train:{}".format(i)) #..
      
      temperature = self.temperature_factor ** i
      _, new_xs, loss = sess.run((self.train_op,
                                  self.trans_op.x_hat,
                                  self.loss),
                                 feed_dict={
                                   self.trans_op.x : xs,
                                   self.trans_op.temperature : temperature
                                 })
      xs = new_xs
      total_loss += loss

    total_loss /= self.step_size
    return total_loss


  def generate(self, sess, generate_size):
    xs = np.random.normal(loc=(-0.11, 0.0),
                          scale=(0.48, 0.49),
                          size=(generate_size, 2))
    xs = xs.astype(np.float32)

    xss = []
    mus = []
    log_sigma_sqs = []

    xss.append(xs)

    for i in reversed(range(-self.extra_step_size, self.step_size)):
      if i < 0:
        i = 0
      temperature = self.temperature_factor ** i
      new_xs, mu, log_sigma_sq = sess.run([self.trans_op.x_hat,
                                           self.trans_op.mu,
                                           self.trans_op.log_sigma_sq],
                                          feed_dict={
                                            self.trans_op.x : xs,
                                            self.trans_op.temperature : temperature
                                          })
      xs = new_xs

      xss.append(xs)
      mus.append(mu)
      log_sigma_sqs.append(log_sigma_sq)

    return np.array(xss), np.array(mus), np.array(log_sigma_sqs)
