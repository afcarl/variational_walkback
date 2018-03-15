# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def save_figure(xs, file_path):
  plt.figure()
  plt.ylim([-2.0, 2.0])
  plt.xlim([-2.0, 2.0])
  plt.plot(xs[:,0], xs[:,1], "ro")
  plt.savefig(file_path)
  plt.close()

  
