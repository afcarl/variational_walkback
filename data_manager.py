# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

class DataManager(object):
  def __init__(self):
    data_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    self.data = input_data.read_data_sets(data_file, one_hot=True)

  @property
  def num_train_examples(self):
    return self.data.train.num_examples

  def get_next_train_batch(self, batch_size):
    images,_ = self.data.train.next_batch(batch_size)
    return images

  def get_next_test_batch(self, batch_size):
    images,_ = self.data.test.next_batch(batch_size)
    return images
