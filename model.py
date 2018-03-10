# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class TransitionOperator(object):
  def __init__(self,
               step,
               alpha,
               learning_rate,
               training):

    # TODO: trainingをplaceholderにすること！！

    self._prepare_network(step, alpha, training)
    if training:
      self._prepare_optimizer(step, learning_rate)

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

  def _prepare_optimizer(self, step, learning_rate):
    with tf.variable_scope("transition_op", reuse=tf.AUTO_REUSE):    
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                     "transition_op/bn{}*".format(step))
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      
      self.loss = tf.reduce_mean(-self.log_p)
      
      with tf.control_dependencies(update_ops):
        self.train_op = optimizer.minimize(self.loss)

    
  def _prepare_network(self, step, alpha, training):
    temperature = float(2 ** step)
    
    with tf.variable_scope("transition_op", reuse=tf.AUTO_REUSE):
      self.x = tf.placeholder(tf.float32, shape=[None, 28*28])
      xr = tf.reshape(self.x, [-1, 28, 28, 1])

      # [Encoder]
      c1 = tf.layers.conv2d(xr, filters=16, kernel_size=[5, 5], strides=(2, 2),
                            padding="same",
                            name="c1")
      c1 = tf.layers.batch_normalization(c1, training=training,
                                         name="bn{}/c1".format(step))
      c1 = tf.nn.relu(c1) # (-1, 14, 14, 16)

      c2 = tf.layers.conv2d(c1, filters=32, kernel_size=[5, 5], strides=(2, 2),
                            padding="same",
                            name="c2")
      c2 = tf.layers.batch_normalization(c2, training=training,
                                         name="bn{}/c2".format(step))
      c2 = tf.nn.relu(c2) # (-1, 7, 7, 32)
      c2_flat = tf.layers.flatten(c2) # (-1, 1568)

      # [Bottleneck]
      f1 = tf.layers.dense(c2_flat, 1024, name="fc1")
      with tf.variable_scope("ln{}/fc1".format(step)) as scope:
        f1 = tf.contrib.layers.layer_norm(f1, activation_fn=tf.nn.leaky_relu,
                                          scope=scope)

      f2 = tf.layers.dense(f1, 1024, name="fc2")
      with tf.variable_scope("ln{}/fc2".format(step)) as scope:
        f2 = tf.contrib.layers.layer_norm(f2, activation_fn=tf.nn.leaky_relu,
                                          scope=scope)

      f3 = tf.layers.dense(f2, 1024, name="fc3")
      with tf.variable_scope("ln{}/fc3".format(step)) as scope:
        f3 = tf.contrib.layers.layer_norm(f3, activation_fn=tf.nn.leaky_relu,
                                          scope=scope)

      f4 = tf.layers.dense(f3, 1024, name="fc4")
      with tf.variable_scope("ln{}/fc4".format(step)) as scope:
        f4 = tf.contrib.layers.layer_norm(f4, activation_fn=tf.nn.leaky_relu,
                                          scope=scope)

      f5 = tf.layers.dense(f4, 1024, name="fc5")
      with tf.variable_scope("ln{}/fc5".format(step)) as scope:
        f5 = tf.contrib.layers.layer_norm(f5, activation_fn=tf.nn.leaky_relu,
                                          scope=scope)
      fr5 = tf.reshape(f5, [-1, 7, 7, 32])

      # [Decoder]
      dc1_m = tf.layers.conv2d_transpose(fr5, filters=16,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc1_m")
      dc1_m = tf.layers.batch_normalization(dc1_m, training=training,
                                            name="bn{}/dc1_m".format(step))
      dc1_m = tf.nn.relu(dc1_m) # (-1, 14, 14, 16)
      
      dc2_m = tf.layers.conv2d_transpose(dc1_m, filters=1,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc2_m")

      dc1_s = tf.layers.conv2d_transpose(fr5, filters=16,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc1_s")
      dc1_s = tf.layers.batch_normalization(dc1_s, training=training,
                                            name="bn{}/dc1_s".format(step))
      
      dc1_s = tf.nn.relu(dc1_s) # (-1, 14, 14, 16)
      dc2_s = tf.layers.conv2d_transpose(dc1_s, filters=1,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc2_s")

      dc2_m = tf.layers.flatten(dc2_m) # (-1, 28*28)
      dc2_s = tf.layers.flatten(dc2_s) # (-1, 28*28)
      
      mu = alpha * self.x + (1.0 - alpha) * dc2_m
      log_sigma_sq = dc2_s + tf.log(temperature)

      x_hat = self._sample(mu, log_sigma_sq)
      x_hat = tf.clip_by_value(x_hat, 0.0, 1.0)

      self.x_hat = x_hat
      self.log_p = self._calc_log_likelihood(self.x, mu, log_sigma_sq)
      

class VariationalWalkback(object):
  def __init__(self,
               alpha=0.5, # 大きいほど元のxに近づける
               learning_rate=1e-4,
               step_size=20,
               training=True):

    self.step_size = step_size
    self._prepare(alpha, learning_rate, training)

  def _prepare(self, alpha, learning_rate, training):
    print("start preparing network")
    
    self.transition_ops = []
    
    for i in range(self.step_size):
      op = TransitionOperator(i, alpha, learning_rate, training)
      self.transition_ops.append(op)

    print("end preparing network")
      

  def train(self, sess, images):
    xs = images
    cur_temperature = 1.0

    total_loss = 0

    for i in range(self.step_size):
      op = self.transition_ops[i]
      _, new_xs, loss = sess.run((op.train_op,
                                  op.x_hat,
                                  op.loss),
                                 feed_dict={
                                   op.x : xs
                                 })
      xs = new_xs
      total_loss += loss

    return total_loss

  """
  def generate(self, sess, generate_size):
    cur_temperature = 2.0 ** 20

    xs = np.random.normal(0.5, 2.0, size=(generate_size, 28*28)).clip(0.0, 1.0)
    for i in range(20):
      new_xs = sess.run(self.x_hat,
                        feed_dict={
                          self.x : xs,
                          self.temperature : cur_temperature
                        })
      cur_temperature /= 2.0
      if cur_temperature <= 1.0:
        cur_temperature = 1.0
      xs = new_xs

    return xs
  """
