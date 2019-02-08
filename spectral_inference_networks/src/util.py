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

"""Utilities for SpIN - mostly plotting, logging and network building."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import tensorflow as tf


def create_default_stats(iterations, neig):
  """Default stats creation function to be passed in as stats hook.

  To be passed in to training loop and called back once the number of Eigen
  values have been determined.

  Args:
    iterations: Number of iterations in training loop.
    neig: Number of Eigen values to track.

  Returns:
    Dict of numpy arrays keyed by stat name.
  """
  losses = np.zeros((iterations), dtype=np.float32)
  eigenvalues = np.zeros((iterations, neig), dtype=np.float32)
  eigenvalues_ma = np.zeros((iterations, neig), dtype=np.float32)
  return {
      'losses': losses,
      'eigenvalues': eigenvalues,
      'eigenvalues_ma': eigenvalues_ma,
  }


def update_default_stats(t, current_loss, current_eigenvalues, losses,
                         eigenvalues, eigenvalues_ma):
  """Update callback for the default stats created above.

  To be passed into training loop and called back once per training step.
  Updates total collections with stats from specified training step.

  Args:
    t: Training step index.
    current_loss: Loss at training step `t`.
    current_eigenvalues: Eigen values at training step `t`.
    losses: Collection of all losses, to be updated at index `t`.
    eigenvalues: Collection of all Eigen values, to be updated at index `t`.
    eigenvalues_ma: Collection of moving averages for Eigen values, to be
      updated at index `t`.
  """
  losses[t] = current_loss
  eigenvalues[t] = current_eigenvalues
  decay = 0.01
  if t > 0:
    eigenvalues_ma[t] = (
        decay * current_eigenvalues + (1 - decay) * eigenvalues_ma[t - 1])
  else:
    eigenvalues_ma[t] = current_eigenvalues


@tf.custom_gradient
def _my_softplus(x):
  def grad(dy):
    return tf.nn.sigmoid(x) * dy
  return tf.nn.softplus(x), grad


def _add_mask(x, y, lim):
  """Makes boundary conditions for network (fixed box)."""
  # Force the wavefunction to zero at the boundaries of the box defined by
  # [-lim, lim].
  mask = 1.0
  for i in range(x.shape.as_list()[1]):
    mask *= tf.maximum((tf.sqrt(2 * lim**2 - x[:, i]**2) - lim) / lim, 0)
  return tf.expand_dims(mask, -1) * y


def make_network(x, hid, ws, bs, apply_boundary, lim, custom_softplus=False):
  """Constructs network and loss function.

  Args:
    x: Input to the network.
    hid: List of shapes of the hidden layers of the networks.
    ws: List of weights of the network.
    bs: List of biases of the network.
    apply_boundary: If true, force network output to be zero at boundary.
    lim: The limit of the network, if apply_boundary is true.
    custom_softplus (optional):

  Returns:
    Output of multi-layer perception network.
  """
  inp = x
  my_softplus = _my_softplus if custom_softplus else tf.nn.softplus

  for i in range(len(hid)-1):
    inp = my_softplus(tf.matmul(inp, ws[i]) + bs[i])

  y = tf.matmul(inp, ws[-1]) + bs[-1]
  if apply_boundary:
    return _add_mask(x, y, lim)
  else:
    return y


def make_conv_network(x, conv_stride, paddings, ws, bs):
  """Creates convolutional network.

  Args:
    x: Input to the convnet.
    conv_stride: List of strides of the convolutions, one per layer.
    paddings: List of paddings of the convolutions, one per layer.
    ws: List of weights. Conv or fully-connected inferred by shape.
    bs: List of biases.

  Returns:
    Output of convolutional neural network.
  """
  inp = x
  nh = len(ws)
  for i in range(nh-1):
    weight = ws[i]

    if len(weight.shape) == 4:
      stride = conv_stride[i]
      inp = tf.nn.relu(tf.nn.conv2d(inp, weight, [1, stride, stride, 1],
                                    padding=paddings[i]) + bs[i])
      # flatten if this is the last conv layer
      if len(ws[i+1].shape) == 2:
        inp = tf.reshape(inp, [inp.shape[0], np.prod(inp.shape[1:])])

    else:
      inp = tf.nn.relu(tf.matmul(inp, weight) + bs[i])

  features = tf.matmul(inp, ws[-1]) + bs[-1]

  dim0 = tf.shape(inp)[0]
  const_feature = tf.ones((dim0, 1))
  features = tf.concat((const_feature, features), 1)

  return features


def grid_reader(dim, lim, points=128):
  """Creates a reader function for generating a grid of position vectors.

  Args:
    dim: Dimension of position vector.
    lim: Limit of the cell. Each vector component is in [-lim, lim].
    points: Number of points to generate along each axis.

  Returns:
    A tensorflow op containing a constant grid of the n-dim box defined by
    [-lim, lim] along each axis. A 2D plane defined by hyperplane is generated
    for n>2-D systems.

  Raises:
    ValueError: len(hyperplane) + 2 != ndim.
  """
  hyperplane = [0 for _ in range(dim - 2)]
  if len(hyperplane) + 2 != dim:
    raise ValueError('Incorrect number of hyperplane values specified.')
  xx = np.linspace(-lim, lim, points, dtype=np.float32)
  yy = np.linspace(-lim, lim, points, dtype=np.float32)
  if dim == 1:
    grid = xx
  elif dim == 2:
    grid = np.meshgrid(xx, yy)
  else:
    zz = [np.linspace(z_i, z_i, 1, dtype=np.float32) for z_i in hyperplane]
    grid = np.meshgrid(xx, yy, *zz)
  xyz = np.array(grid).T.reshape(-1, dim)
  return tf.constant(xyz)
