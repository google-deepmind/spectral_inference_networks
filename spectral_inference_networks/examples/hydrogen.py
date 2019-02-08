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

"""2D Hydrogen atom example from SpIN paper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import matplotlib.pyplot as plt
import numpy as np
import spectral_inference_networks as spin
import tensorflow as tf


flags.DEFINE_integer('neig', 9, 'Number of eigenvalues to compute')
flags.DEFINE_integer('niter', 1000000, 'Number of iterations')
flags.DEFINE_integer('ndim', 2, 'Dimension of space')
flags.DEFINE_integer('batch_size', 128, 'Self-explanatory')
flags.DEFINE_float('laplacian_eps', 0.0,
                   'Finite difference step for Laplacian Operator.')
flags.DEFINE_integer('log_image_every', 10,
                     'Write image of wavefn to log after this many iterations')
flags.DEFINE_integer('save_params_every', 50000,
                     'Save parameters to checkpoint after this many iterations')
flags.DEFINE_float('lim', 50.0, 'Limit of box')
flags.DEFINE_float('lr', 1e-5, 'Learning rate')
flags.DEFINE_float('decay', 0.01, 'Decay rate of moving averages')
flags.DEFINE_boolean('boundary', True, 'Force zero boundary condition')
flags.DEFINE_boolean('show_plots', True,
                     'Show pyplot plots. 2D slices at z=0 are used for ndim=3.')
flags.DEFINE_boolean('use_pfor', True, 'Use parallel_for.')
flags.DEFINE_boolean(
    'per_example', False,
    'Use a different strategy for computing covariance Jacobian')
flags.DEFINE_float('charge', 1.0, 'Nuclear charge of atom.')


FLAGS = flags.FLAGS


def train(iterations,
          batch_size,
          lr,
          ndim=2,
          apply_boundary=False,
          neig=1,
          decay=0.01,
          laplacian_eps=0.1,
          lim=20,
          log_image_every=50000,
          save_params_every=50000,
          show_plots=False,
          use_pfor=False,
          per_example=False,
          charge=0.5):
  """Configures and runs training loop."""

  logging_config = {
      'config': {
          'lr': lr,
          'decay': decay,
          'batch_size': batch_size,
      },
      'log_image_every': log_image_every,
      'save_params_every': save_params_every,
      'saver_path': '/tmp',
      'saver_name': 'hydrogen_params',
  }

  npts = 128
  def _create_plots():
    """Hook to set up plots at start of run."""
    nfig = max(2, int(np.ceil(np.sqrt(neig))))
    psi_fig, psi_ax = plt.subplots(nfig, nfig, figsize=(10, 10))
    psi_im = []
    for i in range(nfig**2):
      psi_ax[i // nfig, i % nfig].axis('off')
    for i in range(neig):
      psi_im.append(psi_ax[i // nfig, i % nfig].imshow(
          np.zeros((npts, npts)), interpolation='none', cmap='plasma'))
    _, loss_ax = plt.subplots(1, 1)
    return psi_fig, psi_ax, psi_im, loss_ax

  def _update_plots(t, outputs, inputs, psi_fig, psi_ax, psi_im, loss_ax,
                    losses=None, eigenvalues=None, eigenvalues_ma=None):
    """Hook to update the plots periodically."""
    del inputs
    del losses
    del eigenvalues
    nfig = max(2, int(np.ceil(np.sqrt(neig))))
    loss_ax.cla()
    loss_ax.plot(eigenvalues_ma[:t])
    if ndim == 2:
      # E(n;Z) = - Z^2 / [2*(n+1/2)^2]
      # Quantum numbers: n=0, 1, ...; m_l = -n, -n+1, ... n
      # degeneracy: 2n+1. Use k^2 as an upper bound to \sum 2n+1.
      max_n = int(np.ceil(np.sqrt(neig))) + 1
      tmp = []
      for n in range(0, max_n):
        for _ in range(2 * n + 1):
          tmp.append(n)
      quantum_nos = np.array(tmp)
      ground_truth = -charge**2 / (2*(quantum_nos[:neig] + 0.5)**2)
    elif ndim == 3:
      # E(n;Z) = - Z^2 / (2n^2)
      # Quantum numbers: n=1, 2, ...; l = 0, 1, ..., n-1; m_l = -l, -l+1, ... l
      # degeneracy: n^2. Use k^3 as an upper bound to \sum n^2.
      max_n = int(np.ceil(neig**(1./3))) + 1
      tmp = []
      for n in range(1, max_n):
        for _ in range(n * n):
          tmp.append(n)
      quantum_nos = np.array(tmp)
      ground_truth = - charge**2 / (2*quantum_nos[:neig]**2)
    ground_truth /= 2.0   # convert back to units in the paper
    for i in range(neig):
      loss_ax.plot([0, t], [ground_truth[i], ground_truth[i]], '--')
    loss_ax.set_ylim([1.0, ground_truth[0]-1])

    for i in range(neig):
      pimg = outputs[:, i].reshape(npts, npts)
      psi_im[i].set_data(pimg)
      psi_im[i].set_clim(pimg.min(), pimg.max())
      psi_ax[i//nfig, i%nfig].set_title(eigenvalues_ma[t, i])
    psi_fig.canvas.draw()
    psi_fig.canvas.flush_events()

  plotting_hooks = {
      'create': _create_plots,
      'update': _update_plots,
  }

  stats_hooks = {
      'create': spin.util.create_default_stats,
      'update': spin.util.update_default_stats,
  }

  k = neig
  hid = (64, 64, 64, k)
  h_ = ndim
  ws = []
  bs = []
  for h in hid:
    ws.append(tf.Variable(tf.random_normal([h_, h])/tf.sqrt(float(h_))))
    bs.append(tf.Variable(tf.random_normal([h])))
    h_ = h
  params = ws + bs

  def network_builder(x):
    return spin.util.make_network(x, hid, ws, bs, apply_boundary, lim,
                                  custom_softplus=not per_example)

  if laplacian_eps == 0.0:
    kinetic = -spin.ExactLaplacianOperator()
  else:
    kinetic = -spin.LaplacianOperator(eps=laplacian_eps)
  potential = spin.DiagonalOperator(
      lambda x: -charge / tf.norm(x, axis=1, keepdims=True))
  hamiltonian = kinetic + potential

  global_step = tf.Variable(0.0, trainable=False)
  optim = tf.train.RMSPropOptimizer(lr, decay=0.999)
  data_for_plotting = spin.util.grid_reader(ndim, lim, npts)

  data = tf.random_uniform([batch_size, ndim], minval=-lim, maxval=lim)

  spectral_net = spin.SpectralNetwork(
      hamiltonian,
      network_builder,
      data,
      params,
      decay=decay,
      use_pfor=use_pfor,
      per_example=per_example)

  stats = spectral_net.train(
      optim,
      iterations,
      logging_config,
      stats_hooks,
      plotting_hooks=plotting_hooks,
      show_plots=show_plots,
      global_step=global_step,
      data_for_plotting=data_for_plotting)

  return stats


def main(argv):
  del argv

  if FLAGS.per_example and FLAGS.laplacian_eps == 0.0:
    raise ValueError('Exact Laplacian is incompatible '
                     'with per-example Jacobian')

  train(
      iterations=FLAGS.niter,
      batch_size=FLAGS.batch_size,
      lr=FLAGS.lr,
      ndim=FLAGS.ndim,
      apply_boundary=FLAGS.boundary,
      neig=FLAGS.neig,
      decay=FLAGS.decay,
      laplacian_eps=FLAGS.laplacian_eps,
      lim=FLAGS.lim,
      log_image_every=FLAGS.log_image_every,
      save_params_every=FLAGS.save_params_every,
      show_plots=FLAGS.show_plots,
      use_pfor=FLAGS.use_pfor,
      per_example=FLAGS.per_example,
      charge=FLAGS.charge)


if __name__ == '__main__':
  app.run(main)
