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
               learning_rate):

    print("creating transition op{}".format(step))
    
    self._prepare_network(step, alpha)
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
    local_scope_name = "transition_op_local{}".format(step)
    
    with tf.variable_scope(local_scope_name):
      # Adamの内部パラメータは各ステップで現在共有していない
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                     local_scope_name)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      
      self.loss = tf.reduce_mean(-self.log_p)
      
      with tf.control_dependencies(update_ops):
        self.train_op = optimizer.minimize(self.loss)

    
  def _prepare_network(self, step, alpha, temperature_rate=1.1):
    temperature = float(temperature_rate ** step)

    reuse_global = step > 0

    local_scope_name = "transition_op_local{}".format(step)
    global_scope_name= "transition_op_global"
    
    with tf.variable_scope(local_scope_name):
      self.training = tf.placeholder(tf.bool, shape=(), name="training")
      self.x = tf.placeholder(tf.float32, shape=[None, 28*28], name="x")

    # [Encoder]
    with tf.variable_scope(global_scope_name, reuse=reuse_global):
      xr = tf.reshape(self.x, [-1, 28, 28, 1])
      c1 = tf.layers.conv2d(xr, filters=16, kernel_size=[5, 5], strides=(2, 2),
                            padding="same",
                            name="c1")

    with tf.variable_scope(local_scope_name):
      c1 = tf.layers.batch_normalization(c1, training=self.training,
                                         name="c1_b".format(step))
        
    with tf.variable_scope(global_scope_name, reuse=reuse_global):
      c1 = tf.nn.relu(c1) # (-1, 14, 14, 16)

      c2 = tf.layers.conv2d(c1, filters=32, kernel_size=[5, 5], strides=(2, 2),
                            padding="same",
                            name="c2")
      
    with tf.variable_scope(local_scope_name):
      c2 = tf.layers.batch_normalization(c2, training=self.training,
                                         name="c2_b".format(step))

    with tf.variable_scope(global_scope_name, reuse=reuse_global):
      c2 = tf.nn.relu(c2) # (-1, 7, 7, 32)
      c2_flat = tf.layers.flatten(c2) # (-1, 1568)

      # [Bottleneck]      
      f1 = tf.layers.dense(c2_flat, 1024, name="fc1")
      
    with tf.variable_scope(local_scope_name + "/ln1") as scope:
      f1 = tf.contrib.layers.layer_norm(f1, activation_fn=tf.nn.leaky_relu,
                                        scope=scope)
      
    with tf.variable_scope(global_scope_name, reuse=reuse_global):
      f2 = tf.layers.dense(f1, 1024, name="fc2")
      
    with tf.variable_scope(local_scope_name + "/ln2") as scope:
      f2 = tf.contrib.layers.layer_norm(f2, activation_fn=tf.nn.leaky_relu,
                                        scope=scope)

    with tf.variable_scope(global_scope_name, reuse=reuse_global):
      f3 = tf.layers.dense(f2, 1024, name="fc3")

    with tf.variable_scope(local_scope_name + "/ln3") as scope:
      f3 = tf.contrib.layers.layer_norm(f3, activation_fn=tf.nn.leaky_relu,
                                        scope=scope)

    with tf.variable_scope(global_scope_name, reuse=reuse_global):
      f4 = tf.layers.dense(f3, 1024, name="fc4")

    with tf.variable_scope(local_scope_name + "/ln4") as scope:
      f4 = tf.contrib.layers.layer_norm(f4, activation_fn=tf.nn.leaky_relu,
                                          scope=scope)

    with tf.variable_scope(global_scope_name, reuse=reuse_global):
      f5 = tf.layers.dense(f4, 1568, name="fc5")

    with tf.variable_scope(local_scope_name + "/ln5") as scope:      
      f5 = tf.contrib.layers.layer_norm(f5, activation_fn=tf.nn.leaky_relu,
                                        scope=scope)

    with tf.variable_scope(global_scope_name, reuse=reuse_global):
      fr5 = tf.reshape(f5, [-1, 7, 7, 32])

      # [Decoder]
      dc1_m = tf.layers.conv2d_transpose(fr5, filters=16,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc1_m")
    with tf.variable_scope(local_scope_name):
      dc1_m = tf.layers.batch_normalization(dc1_m, training=self.training,
                                            name="dc1_m_b".format(step))

    with tf.variable_scope(global_scope_name, reuse=reuse_global):
      dc1_m = tf.nn.relu(dc1_m) # (-1, 14, 14, 16)
      
      dc2_m = tf.layers.conv2d_transpose(dc1_m, filters=1,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc2_m")

      dc1_s = tf.layers.conv2d_transpose(fr5, filters=16,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc1_s")

    with tf.variable_scope(local_scope_name):
      dc1_s = tf.layers.batch_normalization(dc1_s, training=self.training,
                                            name="dc1_s_b".format(step))

    with tf.variable_scope(global_scope_name, reuse=reuse_global):
      dc1_s = tf.nn.relu(dc1_s) # (-1, 14, 14, 16)
      dc2_s = tf.layers.conv2d_transpose(dc1_s, filters=1,
                                         kernel_size=[5,5], strides=(2,2),
                                         padding="same",
                                         name="dc2_s")

      dc2_m = tf.layers.flatten(dc2_m) # (-1, 28*28)
      dc2_s = tf.layers.flatten(dc2_s) # (-1, 28*28)
      
      mu = alpha * self.x + (1.0 - alpha) * dc2_m
      log_sigma_sq = dc2_s + tf.log(temperature)

    with tf.variable_scope(local_scope_name):      
      x_hat = self._sample(mu, log_sigma_sq)
      x_hat = tf.clip_by_value(x_hat, 0.0, 1.0)

      self.x_hat = x_hat
      self.log_p = self._calc_log_likelihood(self.x, mu, log_sigma_sq)
      # 分析用に保存
      self.mu = mu      
      self.log_sigma_sq = log_sigma_sq

      

class VariationalWalkback(object):
  def __init__(self,
               alpha=0.5, # 大きいほど元のxに近づける
               learning_rate=1e-4,
               step_size=30,
               extra_step_size=10):
    
    self.step_size = step_size
    self.extra_step_size = extra_step_size
    self._prepare(alpha, learning_rate)

  def _prepare(self, alpha, learning_rate):
    print("start preparing network")
    
    self.transition_ops = []
    
    for i in range(self.step_size):
      op = TransitionOperator(i, alpha, learning_rate)
      self.transition_ops.append(op)

    print("end preparing network")
      
  def train(self, sess, images):
    xs = images
    total_loss = 0

    for i in range(self.step_size):
      op = self.transition_ops[i]
      _, new_xs, loss = sess.run((op.train_op,
                                  op.x_hat,
                                  op.loss),
                                 feed_dict={
                                   op.x : xs,
                                   op.training : True
                                 })
      xs = new_xs
      total_loss += loss

    total_loss /= self.step_size
    return total_loss

  def generate(self, sess, generate_size):
    xs = np.random.normal(0.5, 2.0, size=(generate_size, 28*28)).clip(0.0, 1.0)
    xs = xs.astype(np.float32)

    mus = []    
    log_sigma_sqs = []

    for i in reversed(range(-self.extra_step_size, self.step_size)):
      if i < 0:
        i = 0
      op = self.transition_ops[i]
      new_xs, mu, log_sigma_sq = sess.run([op.x_hat, op.mu, op.log_sigma_sq],
                                          feed_dict={
                                            op.x : xs,
                                            op.training : False
                                          })
      xs = new_xs
      mus.append(mu)
      log_sigma_sqs.append(log_sigma_sq)

    return xs, np.array(mus), np.array(log_sigma_sqs)
