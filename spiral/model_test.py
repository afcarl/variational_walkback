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
    trans_op = TransitionOperator(alpha)
    # TODO: shape check

    
class VariationalWalkbackTest(tf.test.TestCase):
  def get_batch_data(self, batch_size):
    batch_data = np.zeros((batch_size, 2), dtype=np.float32)
    return batch_data


  def test_train(self):
    model = VariationalWalkback(step_size=2, extra_step_size=2)
    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      data = self.get_batch_data(10)
      loss = model.train(sess, data)

      self.assertEqual(loss.shape, ())


  def test_geneate(self):
    model = VariationalWalkback(step_size=2, extra_step_size=2)
    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      data, mus, sigmas = model.generate(sess, 10)

      self.assertEqual(data.shape,   (4+1, 10, 2))
      self.assertEqual(mus.shape,    (4, 10, 2))
      self.assertEqual(sigmas.shape, (4, 10, 2))

    
if __name__ == "__main__":
  tf.test.main()
