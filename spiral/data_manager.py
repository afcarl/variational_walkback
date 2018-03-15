# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fuel.datasets.toy import Spiral
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

class DataManager(object):
  def __init__(self, data_size, batch_size):
    self.data_set = Spiral(num_examples=data_size,
                           classes=1,
                           cycles=1.0,
                           noise=0.01,
                           sources=('features',))
    self.data_stream = DataStream.default_stream(self.data_set,
                                                 iteration_scheme=ShuffledScheme(
                                                   self.data_set.num_examples,
                                                   batch_size))

    self.all_data_stream = DataStream.default_stream(self.data_set,
                                                      iteration_scheme=ShuffledScheme(
                                                        self.data_set.num_examples,
                                                        self.data_set.num_examples))

  @property
  def num_train_examples(self):
    return self.data_set.num_examples

  def get_next_train_batch(self):
    xbatch = next(self.data_stream.get_epoch_iterator())[0]
    return xbatch

  def get_check_batch(self):
    xbatch = next(self.all_data_stream.get_epoch_iterator())[0]
    return xbatch

