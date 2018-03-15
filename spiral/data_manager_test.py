# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import gc

import utils
from data_manager import DataManager


class DataManagerTest(unittest.TestCase):
  def setUp(self):
    self.manager = DataManager(20000, 5000)

  def tearDown(self):
    del self.manager
    gc.collect()

  def test_num_train_examples(self):
    train_sample_size = self.manager.num_train_examples
    self.assertEqual(train_sample_size, 20000)
  
  def test_get_next_train_batch(self):
    xbatch = self.manager.get_next_train_batch()
    self.assertTrue(xbatch.shape == (5000,2))

    """
    mx = np.mean(xbatch[:,0])
    my = np.mean(xbatch[:,1])
    print("mean={}, {}".format(mx, my))

    vx = np.var(xbatch[:,0])
    vy = np.var(xbatch[:,1])
    print("variance={}, {}".format(vx, vy))
    """

  def test_get_check_batch(self):
    xbatch = self.manager.get_check_batch()
    self.assertTrue(xbatch.shape == (20000,2))
    #utils.save_figure(xbatch, "check.png")
    
    
    

if __name__ == '__main__':
  unittest.main()
