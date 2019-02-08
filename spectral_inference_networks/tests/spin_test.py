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

"""Tests for spectral_inference_networks.spin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import spectral_inference_networks as spin
import tensorflow as tf


class SpinTest(tf.test.TestCase):

  def _small_matrix(self, size=5, neig=3, batch_size=1000, niter=2000,
                    deterministic=False, use_pfor=False, per_example=False):
    """Test SpIN on small matrix."""
    tf.set_random_seed(0)
    np.random.seed(0)

    global_step = tf.Variable(0.0, trainable=False)
    update_global_step = tf.assign(global_step, global_step+1)

    mat = np.random.randn(size, size).astype(np.float32)
    xx = np.dot(mat.transpose(), mat)  # Symmetrize the matrix

    params = [tf.Variable(tf.random_normal([size, neig]))]

    if deterministic:
      decay = 0.0
      # Data is all combinations of rows and columns of the matrix.
      data = tf.concat((tf.tile(tf.eye(size), (size, 1)),
                        tf.reshape(tf.tile(tf.eye(size), (1, size)),
                                   (size**2, size))), axis=0)
      optim = tf.train.GradientDescentOptimizer(1.0)
    else:
      decay = 0.9
      data = tf.one_hot(tf.cast(tf.floor(
          tf.random_uniform([batch_size]) * size), tf.int32), size)
      optim = tf.train.GradientDescentOptimizer(1.0 / global_step)
    data *= np.sqrt(size)  # Make rows unit norm

    def _network(x):
      return tf.matmul(x, params[0])

    def _kernel(x1, x2):
      return tf.reduce_sum(x1 * tf.matmul(x2, xx), axis=1, keepdims=True)

    operator = spin.KernelOperator(_kernel)
    spec_net = spin.SpectralNetwork(
        operator, _network, data, params, decay=decay, use_pfor=use_pfor,
        per_example=per_example)
    step = optim.apply_gradients(zip(spec_net.gradients, params))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      sess.run(update_global_step)
      for _ in range(niter):
        sess.run(step)
      if deterministic:
        eigvec, eigval = sess.run([spec_net.features, spec_net.eigenvalues])
      else:
        eigval = np.zeros(neig)
        n = 1000
        for _ in range(n):
          eigval += sess.run(spec_net.eigenvalues)
        eigval /= n
        eigvec = sess.run(params[0])

    eigvec, _ = np.linalg.qr(eigvec)
    eigvec = eigvec[:int(size)]
    true_eigval, true_eigvec = np.linalg.eig(xx)
    idx = np.argsort(true_eigval)
    print(eigval)
    print(np.sort(true_eigval)[:neig])
    if deterministic:
      atol = 1e-5
    else:
      atol = 1e-1  # Stochastic case is quite noisy
    np.testing.assert_allclose(eigval, np.sort(true_eigval)[:neig], atol=atol)

    # Compute dot product between true eigenvectors and learned ones.
    cross_cov = np.dot(eigvec.transpose(), true_eigvec[:, idx[:neig]])
    cross_cov -= np.diag(np.diag(cross_cov))
    np.testing.assert_allclose(cross_cov, np.zeros((neig, neig)), atol=atol)

  def test_small_matrix_stochastic_use_pfor_false_per_example_false(self):
    self._small_matrix(deterministic=False, use_pfor=False, per_example=False)

  def test_small_matrix_stochastic_use_pfor_true_per_example_false(self):
    self._small_matrix(deterministic=False, use_pfor=True, per_example=False)

  def test_small_matrix_stochastic_use_pfor_true_per_example_true(self):
    self._small_matrix(deterministic=False, use_pfor=True, per_example=True)

  def test_small_matrix_deterministic_use_pfor_false_per_example_false(self):
    self._small_matrix(deterministic=True, use_pfor=False, per_example=False)

  def test_small_matrix_deterministic_use_pfor_true_per_example_false(self):
    self._small_matrix(deterministic=True, use_pfor=True, per_example=False)

  def test_small_matrix_deterministic_use_pfor_true_per_example_true(self):
    self._small_matrix(deterministic=True, use_pfor=True, per_example=True)


if __name__ == '__main__':
  tf.test.main()
