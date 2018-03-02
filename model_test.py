# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model import VariationalWalkback


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




if __name__ == "__main__":
  tf.test.main()
