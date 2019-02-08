# Copyright 2018-2019 DeepMind Technologies Limited and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo of SpIN on Atari dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging

import matplotlib.pyplot as plt
import numpy as np
import spectral_inference_networks as spin
import tensorflow as tf

EXAMPLES_ROOT = os.path.dirname(__file__)

flags.DEFINE_integer(
    'neig', 5, 'Number of Eigen values to compute. Must be greater than 1.')
flags.DEFINE_integer('niter', 100000, 'Number of iterations.')
flags.DEFINE_integer('batch_size', 128, 'Self-explanatory.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('decay', 0.01, 'Decay rate of moving averages.')
flags.DEFINE_float('rmsprop_decay', 0.1, 'Decay param for RMSprop.')
flags.DEFINE_boolean('step_lr', False, 'Step down learning rate exponentially.')
flags.DEFINE_boolean('show_plots', True, 'Show pyplot plots.')
flags.DEFINE_boolean('use_pfor', True, 'Use parallel_for.')
flags.DEFINE_boolean(
    'per_example', True,
    'Use a different strategy for computing covariance jacobian.')
flags.DEFINE_string(
    'data_dir', None, 'Directory to load game data from. If unspecified, this '
    'will default to the enclosed example data.')
flags.DEFINE_string('game', 'montezuma_revenge',
                    '(montezuma_revenge|space_invaders|beam_rider).')
flags.DEFINE_integer('log_image_every', 10,
                     'No need to write images for this experiment.')
flags.DEFINE_integer(
    'save_params_every', 50000,
    'Save parameters to checkpoint after this many iteration.')
flags.DEFINE_integer('shards', 50, 'Number of shards to load, for speed.')

FLAGS = flags.FLAGS

_NFRAMES = 10000
_IMAGE_SIZE = 84  # Image side length.
_KERNEL_SIZE = 6


def train(iterations,
          lr,
          batch_size,
          neig,
          shards,
          game,
          step_lr=False,
          decay=0.01,
          rmsprop_decay=0.1,
          log_image_every=10,
          save_params_every=50000,
          use_pfor=False,
          per_example=False,
          data_dir=None,
          show_plots=False):
  """Sets up and starts training for SpIN on Atari video data."""

  if data_dir is None:
    data_dir = os.path.join(EXAMPLES_ROOT, 'atari_episodes')

  conv_size = [64, 64, 64]  # number of channels in each conv layer
  conv_stride = [2, 2, 2]  # stride of each conv layer
  # number of units in fully connected layers
  fc_size = [6400, 128, neig]
  paddings = ['VALID', 'SAME', 'SAME']
  nc_ = 4  # initial number of channels
  ws = []
  bs = []

  for nc in conv_size:
    stddev = 1 / np.sqrt(nc_ * _KERNEL_SIZE**2)
    ws.append(
        tf.Variable(
            tf.truncated_normal([_KERNEL_SIZE, _KERNEL_SIZE, nc_, nc],
                                stddev=stddev)))
    bs.append(tf.Variable(tf.zeros([nc])))
    nc_ = nc

  for i in range(1, len(fc_size)):
    ws.append(tf.Variable(tf.truncated_normal([fc_size[i-1], fc_size[i]],
                                              stddev=1/np.sqrt(fc_size[i-1]))))
    bs.append(tf.Variable(tf.zeros([fc_size[i]])))

  params = ws + bs
  saver_path = '/tmp'

  logging_config = {
      'config': {
          'lr': lr,
          'decay': decay,
          'batch_size': batch_size,
          'rmsprop_decay': rmsprop_decay,
          'game': game,
      },
      'log_image_every': log_image_every,
      'save_params_every': save_params_every,
      'saver_path': saver_path,
      'saver_name': game + '_params',
  }

  stats_hooks = {
      'create': spin.util.create_default_stats,
      'update': spin.util.update_default_stats,
  }

  def _create_plots():
    """Hook to set up plots at start of run."""
    frame_fig, frame_ax = plt.subplots(2, neig, figsize=(neig * 8, 8))
    frame_im = []

    for i in range(2):
      for j in range(neig):
        frame_ax[i, j].axis('off')
        frame_im.append(frame_ax[i, j].imshow(
            np.zeros((_IMAGE_SIZE, _IMAGE_SIZE)),
            interpolation='none',
            cmap='gray', vmin=0.0, vmax=255.0))

    _, loss_ax = plt.subplots(1, 1)
    return frame_fig, frame_im, loss_ax

  def _update_plots(t,
                    outputs,
                    inputs,
                    frame_fig,
                    frame_im,
                    loss_ax,
                    losses=None,
                    eigenvalues=None,
                    eigenvalues_ma=None):
    """Hook to update the plots periodically."""
    del losses
    del eigenvalues

    for i in range(neig):
      ordered = np.argsort(outputs[:, i+1])  # sort features for this minibatch
      frame_im[i].set_data(inputs[ordered[0], ..., -1])
      frame_im[i+neig].set_data(inputs[ordered[-1], ..., -1])

    frame_fig.canvas.draw()
    frame_fig.canvas.flush_events()

    loss_ax.cla()
    loss_ax.plot(eigenvalues_ma[:t])

    if t > 0:
      ymin = eigenvalues_ma[max(0, t-1000):t].min()
      ymax = eigenvalues_ma[max(0, t-1000):t].max()
      ydiff = ymax - ymin
      loss_ax.set_ylim([ymin-0.1*ydiff, ymax+0.1*ydiff])

  plotting_hooks = {
      'create': _create_plots,
      'update': _update_plots,
  }

  global_step = tf.Variable(0.0, trainable=False)

  def network_builder(x):
    return spin.util.make_conv_network(x, conv_stride, paddings, ws, bs)

  if step_lr:
    lr = tf.train.exponential_decay(lr * decay, global_step, 100 / decay, 0.8)
  optim = tf.train.RMSPropOptimizer(
      lr, decay=(1.0 - decay * rmsprop_decay), centered=True)

  logging.info('Loading game %s', game)
  episodes = np.load(os.path.join(data_dir, '{}.npz'.format(game)))

  frames = episodes['frames']
  episode_starts = episodes['episode_starts']

  batch = np.zeros((batch_size + 1, _IMAGE_SIZE, _IMAGE_SIZE, 4),
                   dtype=np.float32)

  def _reader():
    idx = np.random.randint(0, (_NFRAMES * shards) - batch_size - 4)
    while np.any(episode_starts[idx+1:idx+batch_size+4]):
      idx = np.random.randint(0, (_NFRAMES * shards) - batch_size - 4)
    for i in range(batch_size+1):
      batch[i] = frames[idx+i:idx+i+4].transpose((1, 2, 0))
    return batch

  data = tf.py_func(_reader, [], [tf.float32])[0]
  data.set_shape([batch_size + 1, _IMAGE_SIZE, _IMAGE_SIZE, 4])

  spectral_net = spin.SpectralNetwork(
      spin.SlownessOperator(),
      network_builder,
      data,
      params,
      decay=decay,
      use_pfor=use_pfor,
      per_example=per_example)

  spectral_net.train(
      optim,
      iterations,
      logging_config,
      stats_hooks,
      plotting_hooks=plotting_hooks,
      show_plots=show_plots,
      global_step=global_step,
      data_for_plotting=data)


def main(argv):
  del argv

  if FLAGS.neig < 2:
    raise ValueError('Number of Eigen values must be at least 2.')

  train(
      iterations=FLAGS.niter,
      lr=FLAGS.lr,
      batch_size=FLAGS.batch_size,
      neig=FLAGS.neig,
      shards=FLAGS.shards,
      step_lr=FLAGS.step_lr,
      decay=FLAGS.decay,
      rmsprop_decay=FLAGS.rmsprop_decay,
      game=FLAGS.game,
      log_image_every=FLAGS.log_image_every,
      save_params_every=FLAGS.save_params_every,
      use_pfor=FLAGS.use_pfor,
      per_example=FLAGS.per_example,
      data_dir=FLAGS.data_dir,
      show_plots=FLAGS.show_plots)


if __name__ == '__main__':
  app.run(main)
