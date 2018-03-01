# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model import VariationalWallback


class VariationalWallbackTest(tf.test.TestCase):
  """
  def get_batch_images(self, batch_size):
    image = np.zeros((28*28), dtype=np.float32)
    batch_images = [image] * batch_size
    return batch_images
  """

  def test_init(self):
    model = VariationalWallback()
    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      """
      self.assertEqual(model.loss.get_shape(), ())
      self.assertEqual(model.x_out_mu.get_shape()[1],           2)
      self.assertEqual(model.x_out_log_sigma_sq.get_shape()[1], 2)
      self.assertEqual(model.x_out.get_shape()[1],              2)
      """


if __name__ == "__main__":
  tf.test.main()
