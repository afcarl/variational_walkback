# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model import VariationalWalkback, TransitionOperator


class TransitionOperatorTest(tf.test.TestCase):
  def test_init(self):
    alpha = 0.5
    learning_rate = 1e-4
    
    conv_var_size = 2
    bn_var_size = 4
    fc_var_size = 2
    ln_var_size = 2

    conv_var_size_adam = 2*2
    bn_var_size_adam = 2*2
    fc_var_size_adam = 2*2
    ln_var_size_adam = 2*2

    # Step0でTransitionOperator作成
    trans_op0 = TransitionOperator(0, alpha, learning_rate)
    variables_global0 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope="transition_op_global")
    variables_local0 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope="transition_op_local0")
    
    var_size_global0 = conv_var_size*6 + fc_var_size*5
    var_size_local0  = bn_var_size*4 + ln_var_size*5 + \
                       conv_var_size_adam*6 + bn_var_size_adam*4 + \
                       fc_var_size_adam*5 + ln_var_size_adam*5 + \
                       + 2 # +2は、beta1_power, beta2_powerの分
    
    #print("len(variables_global0)={}".format(len(variables_global0))) # 22
    #print("len(variables_local0)={}".format(len(variables_local0)))   # 108
    
    # variable数を確認
    self.assertEqual(len(variables_global0), var_size_global0)
    self.assertEqual(len(variables_local0),  var_size_local0)


    # Step1でTransitionOperator作成
    trans_op1 = TransitionOperator(1, alpha, learning_rate)

    variables_global1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope="transition_op_global")
    variables_local1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope="transition_op_local1")
    
    #print("len(variables_global1)={}".format(len(variables_global1))) # 22
    #print("len(variables_local1)={}".format(len(variables_local1)))   # 108

    self.assertEqual(len(variables_global1), var_size_global0)
    self.assertEqual(len(variables_local1),  var_size_local0)


class VariationalWalkbackTest(tf.test.TestCase):
  def get_batch_images(self, batch_size):
    batch_images = np.zeros((batch_size, 28*28), dtype=np.float32)
    return batch_images


  def test_train(self):
    model = VariationalWalkback(step_size=2, extra_step_size=2)
    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      images = self.get_batch_images(10)
      loss = model.train(sess, images)

      self.assertEqual(loss.shape, ())
      

  def test_geneate(self):
    model = VariationalWalkback(step_size=2, extra_step_size=2)
    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      images, mus, log_sigma_sqs = model.generate(sess, 10)

      self.assertEqual(images.shape, (10, 28*28))
      self.assertEqual(mus.shape, (4, 10, 28*28))
      self.assertEqual(log_sigma_sqs.shape, (4, 10, 28*28))

      sigma_sqs = np.exp(log_sigma_sqs)
      for i in range(mus.shape[0]):
        mu       = mus[i]
        sigma_sq = sigma_sqs[i]
        mean_mu       = np.mean(mu)
        mean_sigma_sq = np.mean(sigma_sq)
        print("mu[{0}]       = {1:.2f}".format(i, mean_mu))
        print("sigma_sq[{0}] = {1:.2f}".format(i, mean_sigma_sq))


if __name__ == "__main__":
  tf.test.main()
