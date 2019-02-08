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

"""Tests for the hydrogen example for SpIN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from spectral_inference_networks.examples import hydrogen
import tensorflow as tf


class HydrogenTest(tf.test.TestCase):

  def test_hydrogen(self):
    hydrogen.train(
        iterations=50,
        batch_size=8,
        lr=1e-4,
        apply_boundary=True,
        neig=4)

  def test_hydrogen_exact_lapl(self):
    hydrogen.train(
        iterations=50,
        batch_size=8,
        lr=1e-4,
        apply_boundary=True,
        neig=4,
        laplacian_eps=0.0)

  def test_hydrogen_with_pfor_and_per_example(self):
    hydrogen.train(
        iterations=50,
        batch_size=8,
        lr=1e-4,
        apply_boundary=True,
        neig=4,
        use_pfor=True,
        per_example=True)

  def test_hydrogen_exact_lapl_with_pfor_and_per_example(self):
    hydrogen.train(
        iterations=50,
        batch_size=8,
        lr=1e-4,
        apply_boundary=True,
        neig=4,
        laplacian_eps=0.0,
        use_pfor=True,
        per_example=True)


if __name__ == '__main__':
  tf.test.main()
