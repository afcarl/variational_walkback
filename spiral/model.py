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

  def _sample(self, mu, sigma):
    eps_shape = tf.shape(mu)
    eps = tf.random_normal( eps_shape, 0.0, 1.0, dtype=tf.float32 )
    out = tf.add(mu,
                 tf.multiply(sigma, eps))
    return out

  def _calc_log_likelihood(self, x, mu, sigma):
    log_p = -0.5 * tf.log(2.0 * np.pi) \
            - tf.log(sigma) \
            - tf.square(x - mu) / (2.0 * sigma)
    # Correct denominator should be, (2.0 * tf.square(sigma)) but why?
    return tf.reduce_sum(log_p, 1)

  def _prepare_network(self, alpha, sigma_factor=0.009):
    with tf.variable_scope("transition_op"):
      self.temperature = tf.placeholder(tf.float32, shape=[],        name="temperature")
      self.x           = tf.placeholder(tf.float32, shape=[None, 2], name="x")
      
      h1 = tf.layers.dense(self.x, 4096, name="fc1", activation=tf.nn.relu)
      h2 = tf.layers.dense(h1,     4096, name="fc2", activation=tf.nn.relu)
      
      mu_org    = tf.layers.dense(h2, 2, name="mu")
      sigma_org = tf.layers.dense(h2, 2, name="sigma", activation=tf.nn.softplus)
      
      mu = alpha * self.x + (1.0 - alpha) * mu_org
      sigma = sigma_factor * sigma_org * tf.sqrt(self.temperature)

      x_hat = self._sample(mu, sigma)

      self.x_hat = x_hat
      self.log_p = self._calc_log_likelihood(self.x, mu, sigma)
      
      self.mu = mu
      self.sigma = sigma



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
    sigmas = []

    xss.append(xs)

    for i in reversed(range(-self.extra_step_size, self.step_size)):
      if i < 0:
        i = 0
      temperature = self.temperature_factor ** i
      new_xs, mu, sigma = sess.run([self.trans_op.x_hat,
                                    self.trans_op.mu,
                                    self.trans_op.sigma],
                                   feed_dict={
                                     self.trans_op.x : xs,
                                     self.trans_op.temperature : temperature
                                   })
      xs = new_xs

      xss.append(xs)
      mus.append(mu)
      sigmas.append(sigma)

    return np.array(xss), np.array(mus), np.array(sigmas)
