# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import gc

from data_manager import DataManager


class DataManagerTest(unittest.TestCase):
  def setUp(self):
    self.manager = DataManager()

  def tearDown(self):
    del self.manager
    gc.collect()

  def test_num_train_examples(self):
    train_sample_size = self.manager.num_train_examples
    self.assertEqual(train_sample_size, 55000)
  
  def test_get_next_train_batch(self):
    images = self.manager.get_next_train_batch(10)
    self.assertTrue(images.shape == (10,28*28))

  def test_get_next_test_batch(self):
    images = self.manager.get_next_test_batch(10)
    self.assertTrue(images.shape == (10,28*28))
    

if __name__ == '__main__':
  unittest.main()
