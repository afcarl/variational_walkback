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
    trans_op0 = TransitionOperator(0, alpha, learning_rate, training=True)
    variables0 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope="transition_op")

    for v in variables0:
      print(v.name)
    
    var_size0 = conv_var_size*6 + bn_var_size*4 + fc_var_size*5 + ln_var_size*5 + \
    conv_var_size_adam*6 + bn_var_size_adam*4 + fc_var_size_adam*5 + ln_var_size_adam*5 + \
    + 2 
    # 48+64+2
    # +2は、beta1_power, beta2_powerの分

    # variable数を確認
    self.assertEqual(len(variables0), var_size0)

    # Step1でTransitionOperator作成
    trans_op1 = TransitionOperator(1, alpha, learning_rate, training=True)
    variables1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope="transition_op")
    # Batch Norm, Layer Normの分だけvariable数が増えているのを確認
    var_size1 = var_size0 + \
                bn_var_size*4 + ln_var_size*5 + \
                bn_var_size_adam*4 + ln_var_size_adam*5 + \
                2

    self.assertEqual(len(variables1), var_size1)
    
    # training=Falseで作成
    trans_gen_op0 = TransitionOperator(0, alpha, learning_rate, training=False)
    trans_gen_op1 = TransitionOperator(1, alpha, learning_rate, training=False)

    variables1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope="transition_op")
    self.assertEqual(len(variables1), var_size1)


"""
class VariationalWalkbackTest(tf.test.TestCase):
  def get_batch_images(self, batch_size):
    image = np.zeros((28*28), dtype=np.float32)
    batch_images = [image] * batch_size
    return batch_images


  def test_init(self):
    model = VariationalWalkback()
    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      self.assertEqual(model.loss.get_shape(), ())
      self.assertEqual(model.x_hat.get_shape()[1], 28*28)


  def test_train(self):
    model = VariationalWalkback()
    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      images = self.get_batch_images(10)
      loss = model.train(sess, images)

      self.assertEqual(loss.shape, ())


  def test_reuse(self):
    model0 = VariationalWalkback(reuse=False, training=True)
    model1 = VariationalWalkback(reuse=True, training=False)


  def test_geneate(self):
    model = VariationalWalkback()
    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      images = model.generate(sess, 10)

      self.assertEqual(images.shape, (10, 28*28))
"""



if __name__ == "__main__":
  tf.test.main()
