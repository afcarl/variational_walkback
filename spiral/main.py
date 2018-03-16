# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from model import VariationalWalkback
from data_manager import DataManager
import utils


tf.app.flags.DEFINE_string("save_dir", "saved", "checkpoints,log,options save directory")
tf.app.flags.DEFINE_integer("epoch_size", 3, "epoch size")
tf.app.flags.DEFINE_integer("data_size", 20000, "data size")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_float("alpha", 0.5, "alpha param for transition op mean output")
tf.app.flags.DEFINE_float("learning_rate", 1e-5, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")

flags = tf.app.flags.FLAGS


def train(sess,
          model,
          data_manager,
          saver,
          start_step):

  log_dir = flags.save_dir + "/log"
  summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
  
  n_samples = data_manager.num_train_examples

  step = start_step
  start_epoch = start_step // n_samples
  
  for epoch in range(start_epoch, flags.epoch_size):
    avg_cost = 0.0
    total_batch = n_samples // flags.batch_size
    
    for i in range(total_batch):
      batch_xs = data_manager.get_next_train_batch()
      
      loss = model.train(sess, batch_xs)
      print("loss={}".format(loss))
      #summary_writer.add_summary(summary_str, step)
      
      step += 1
      
    print("generate: epoch={}".format(epoch))
    generate_check(sess, model)

    if epoch % 3 == (3-1):
      # Save checkpoint
      save_checkponts(sess, saver, step)


def generate_check(sess, model):
  xss, mus, sigmas = model.generate(sess, 500)
  
  image_dir = flags.save_dir + "/generated"
  if not os.path.exists(image_dir):
    os.mkdir(image_dir)

  for i in range(len(xss)):
    xs = xss[i]
    file_name = "generated_{0:0>2}.png".format(i)
    file_path = image_dir + "/" + file_name
    utils.save_figure(xs, file_path)

  # Print mu and sigma at each step.
  for i in range(mus.shape[0]):
    mu    = mus[i]
    sigma = sigmas[i]
    mean_mu    = np.mean(mu)
    mean_sigma = np.mean(sigma)
    print("mu[{0}]    = {1:.5f}".format(i, mean_mu))
    print("sigma[{0}] = {1:.5f}".format(i, mean_sigma))



def load_checkpoints(sess):
  saver = tf.train.Saver(max_to_keep=2)
  checkpoint_dir = flags.save_dir + "/checkpoints"
  
  checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    # checkpointからロード
    saver.restore(sess, checkpoint.model_checkpoint_path)
    # ファイル名から保存時のstep数を復元
    tokens = checkpoint.model_checkpoint_path.split("-")
    step = int(tokens[1])
    print("Loaded checkpoint: {0}, step={1}".format(checkpoint.model_checkpoint_path, step))
    return saver, step+1
  else:
    print("Could not find old checkpoint")
    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)
    return saver, 0

def save_checkponts(sess, saver, global_step):
  checkpoint_dir = flags.save_dir + "/checkpoints"
  saver.save(sess, checkpoint_dir + '/' + 'checkpoint', global_step=global_step)
  print("Checkpoint saved")


def main(argv):
  if not os.path.exists(flags.save_dir):
    os.mkdir(flags.save_dir)
  
  data_manager = DataManager(flags.data_size, flags.batch_size)

  sess = tf.Session()

  step_size = 30
  extra_step_size = 10

  model = VariationalWalkback(alpha=flags.alpha,
                              learning_rate=flags.learning_rate,
                              step_size=step_size,
                              extra_step_size=extra_step_size)

  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  sess.run(init_op)
  
  saver, start_step = load_checkpoints(sess)

  train(sess, model, data_manager, saver, start_step)


if __name__ == '__main__':
  tf.app.run()
