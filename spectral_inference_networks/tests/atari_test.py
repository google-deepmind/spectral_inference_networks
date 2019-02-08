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

"""Tests for the Atari example for SpIN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from spectral_inference_networks.examples import atari
import tensorflow as tf


class AtariTest(tf.test.TestCase):

  def test_atari(self):
    atari.train(
        iterations=10,
        batch_size=4,
        lr=1e-4,
        neig=2,
        shards=1,
        game='montezuma_revenge')

  def test_atari_with_per_example_and_pfor(self):
    atari.train(
        iterations=10,
        batch_size=4,
        lr=1e-4,
        neig=2,
        shards=1,
        game='montezuma_revenge',
        use_pfor=True,
        per_example=True)

if __name__ == '__main__':
  tf.test.main()
