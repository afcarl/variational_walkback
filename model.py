# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class VariationalWallback(object):
  def __init__(self,
               training=True,
               learning_rate=1e-4):

    self.training = training
    self.learning_rate = learning_rate

    self._prepare()

  def _prepare(self):
    self.x = tf.placeholder(tf.float32, shape=[None, 28*28])
    self.temperature = tf.placeholder(tf.float32, shape=[])

    self.x_hat, log_p = self._transition_operator(self.x)
    
    self.loss = -log_p

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    
    with tf.control_dependencies(update_ops):
      self.train_op = optimizer.minimize(self.loss)
    
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

  def _transition_operator(self, x):
    with tf.variable_scope("transition_op"):
      xr = tf.reshape(x, [-1, 28, 28, 1])
      
      c1 = tf.layers.conv2d(xr, filters=16, kernel_size=[5, 5], strides=(2, 2),
                            padding="same",
                            name="c1")
      c1 = tf.layers.batch_normalization(c1, training=self.training,
                                         name="c1_b")
      c1 = tf.nn.relu(c1) # (-1, 14, 14, 16)

      c2 = tf.layers.conv2d(c1, filters=32, kernel_size=[5, 5], strides=(2, 2),
                            padding="same",
                            name="c2")
      c2 = tf.layers.batch_normalization(c2, training=self.training,
                                         name="c2_b")
      c2 = tf.nn.relu(c2) # (-1, 7, 7, 32)
      c2_flat = tf.layers.flatten(c2) # (-1, 1568)

      f1 = tf.layers.dense(c2_flat, 1024, name="fc1")
      f1 = tf.contrib.layers.layer_norm(f1, activation_fn=tf.nn.leaky_relu)
      f2 = tf.layers.dense(f1, 1024, name="fc2")
      f2 = tf.contrib.layers.layer_norm(f2, activation_fn=tf.nn.leaky_relu)
      f3 = tf.layers.dense(f2, 1024, name="fc3")
      f3 = tf.contrib.layers.layer_norm(f3, activation_fn=tf.nn.leaky_relu)
      f4 = tf.layers.dense(f3, 1024, name="fc4")
      f4 = tf.contrib.layers.layer_norm(f4, activation_fn=tf.nn.leaky_relu)
      f5 = tf.layers.dense(f4, 1568, name="fc5")
      f5 = tf.contrib.layers.layer_norm(f5, activation_fn=tf.nn.leaky_relu)

      fr5 = tf.reshape(f5, [-1, 7, 7, 32])

      dc1_m = tf.layers.conv2d_transpose(fr5, filters=16,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc1_m")
      dc1_m = tf.layers.batch_normalization(dc1_m, training=self.training,
                                            name="dc1_m_b")
      dc1_m = tf.nn.relu(dc1_m) # (-1, 14, 14, 16)
      dc2_m = tf.layers.conv2d_transpose(dc1_m, filters=1,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc2_m")

      dc1_s = tf.layers.conv2d_transpose(fr5, filters=16,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc1_s")
      dc1_s = tf.layers.batch_normalization(dc1_s, training=self.training,
                                            name="dc1_s_b")
      dc1_s = tf.nn.relu(dc1_s) # (-1, 14, 14, 16)
      dc2_s = tf.layers.conv2d_transpose(dc1_s, filters=1,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc2_s")

      dc2_m = tf.layers.flatten(dc2_m) # (-1, 28*28)
      dc2_s = tf.layers.flatten(dc2_s) # (-1, 28*28)
      
      mu           = dc2_m
      log_sigma_sq = dc2_s + tf.log(self.temperature)

      x_sampled = self._sample(mu, log_sigma_sq)

      x_hat = x * 0.5 + x_sampled * 0.5
      x_hat = tf.stop_gradient(x_hat)
      x_hat = tf.clip_by_value(x_hat, 0.0, 1.0)
      
      log_p = self._calc_log_likelihood(x, mu, log_sigma_sq)
      
      return x_hat, log_p
