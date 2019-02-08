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

"""Package implementing Spectral Inference Networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import logging

import matplotlib.pyplot as plt
from spectral_inference_networks.src import util
import tensorflow as tf
from tensorflow.python.ops import parallel_for as pfor  # pylint: disable=g-direct-tensorflow-import

assert util, ('spectral_inference_networks.src.util must be imported.')


def _collapse_first_dim(x):
  new_shape = tf.concat([[-1], tf.shape(x)[2:]], axis=0)
  return tf.reshape(x, new_shape)


def _objective_grad(xx, obj, grad_loss, grad_eigval, grad_chol):
  """Symbolic form of the gradient of the objective with stop_gradients."""
  del grad_eigval
  del grad_chol
  with tf.name_scope('objective_grad'):
    chol = tf.cholesky(xx)
    choli = tf.linalg.inv(chol)
    rq = tf.matmul(choli, tf.matmul(obj, choli, transpose_b=True))

    dl = tf.diag(tf.matrix_diag_part(choli))
    triu = tf.matrix_band_part(tf.matmul(rq, dl), 0, -1)
    gxx = -1.0 * tf.matmul(choli, triu, transpose_a=True)
    gobj = tf.matmul(choli, dl, transpose_a=True)

    return grad_loss * gxx, grad_loss * gobj


@tf.custom_gradient
def _objective(xx, obj):
  """Objective function as custom op so that we can overload gradients."""
  with tf.name_scope('objective'):
    chol = tf.cholesky(xx)
    choli = tf.linalg.inv(chol)

    rq = tf.matmul(choli, tf.matmul(obj, choli, transpose_b=True))
    eigval = tf.matrix_diag_part(rq)
    loss = tf.trace(rq)
    grad = functools.partial(_objective_grad, xx, obj)
  return (loss, eigval, chol), grad


@tf.custom_gradient
def _covariance(x, y):
  """Covariance function as custom op."""
  with tf.name_scope('covariance'):
    cov = tf.matmul(x, y, transpose_a=True) / tf.cast(tf.shape(x)[0], x.dtype)

  def _cov_grad(grad):
    with tf.name_scope('cov_grad'):
      return (tf.matmul(y, grad) / tf.cast(tf.shape(x)[0], x.dtype),
              tf.matmul(x, grad) / tf.cast(tf.shape(x)[0], x.dtype))

  return cov, _cov_grad


class LinearOperator(object):
  """Base class for different linear operators that can be combined."""

  def __init__(self, op=None):
    self._op = op

  def build_network(self, f, x):
    """Build network from a builder f'n for the network 'f' and data 'x'."""
    self.f = f(x)
    return self.f

  def build_op(self, x, logpdf=None):
    """Build op from data 'x'."""
    del x
    del logpdf
    raise ValueError('build_op not implemented in derived class.')

  def build(self, f, x):
    """Combines build_network and build_op."""
    fx = self.build_network(f, x)
    # For per_example Jacobian computation, the features and the Jacobian
    # of the features must be created at the same time.
    # Note that this only works if:
    #  1) build_network is never called externally
    #  2) build is never overridden by a child class
    if isinstance(fx, tuple):
      self._jac = fx[1]
      self.f = fx[0]
    op = self.build_op(x)
    return self.f, op

  @property
  def op(self):
    return self._op

  @property
  def jac(self):
    # Only exists if we are computing the Jacobian per-example.
    return self._jac

  def __add__(self, x):
    return AddOperator(self, x)

  def __sub__(self, x):
    return AddOperator(self, -x)

  def __mul__(self, c):
    return ScaleOperator(c, self)

  def __rmul__(self, c):
    return ScaleOperator(c, self)

  def __neg__(self):
    return -1*self

  # Comparison operators:
  # Only used to decide order precedence for calling build_network.
  def __lt__(self, x):
    # Only Laplacians or things made of Laplacians take precedence.
    if isinstance(x, ScaleOperator):
      return self < x.x
    if isinstance(x, AddOperator):
      return self < x.x or self < x.y
    return (isinstance(x, LaplacianOperator) or
            isinstance(x, ExactLaplacianOperator))

  def __le__(self, x):
    return True  # only override this for super-classes.

  def __gt__(self, x):
    return False

  def __ge__(self, x):
    return not self < x


class ScaleOperator(LinearOperator):
  """Linear operator formed by scaling."""

  def __init__(self, c, x):
    super(ScaleOperator, self).__init__()
    self.c = c
    self.x = x

  def build_network(self, f, x):
    self.f = self.x.build_network(f, x)
    return self.f

  def build_op(self, x):
    self._op_x = self.x.build_op(x)
    self._op = self.c * self._op_x
    return self._op

  def __lt__(self, x):
    return self.x < x

  def __le__(self, x):
    return self.x <= x

  def __gt__(self, x):
    return self.x > x

  def __ge__(self, x):
    return self.x >= x


class AddOperator(LinearOperator):
  """Linear operator formed by adding two operators together."""

  def __init__(self, x, y):
    super(AddOperator, self).__init__()
    self.x = x
    self.y = y

  def build_network(self, f, x):
    # Use comparison to choose precedence for order of building network.
    if self.x >= self.y:
      self.f = self.x.build_network(f, x)
      self.y.f = self.f
    else:
      self.f = self.y.build_network(f, x)
      self.x.f = self.f
    return self.f

  def build_op(self, x, logpdf=None):
    self._op_x = self.x.build_op(x)
    self._op_y = self.y.build_op(x)
    self._op = self._op_x + self._op_y
    return self._op

  def __lt__(self, x):
    return self.x < x and self.y < x

  def __le__(self, x):
    return self.x <= x and self.y <= x

  def __gt__(self, x):
    return not self <= x


class LaplacianOperator(LinearOperator):
  """Finite difference Laplacian operator."""

  def __init__(self, eps):
    super(LaplacianOperator, self).__init__()
    self._eps = eps

  def _perturbation(self, x, eps):
    ndim = x.shape.as_list()[1]  # dimension of position vector (i.e. 1,2,3).
    xs = [x]
    for i in range(ndim):
      xs.append(x + eps * tf.one_hot(i, ndim))
      xs.append(x - eps * tf.one_hot(i, ndim))
    return tf.concat(xs, axis=0)

  def build_network(self, f, x):
    """Build operator from a builder f'n for the network 'f' and data 'x'."""
    xs = self._perturbation(x, self._eps)
    fx = f(xs)  # build network, then return it at the end.
    ndim = x.shape.as_list()[1]

    if isinstance(fx, tuple):
      jac = [tf.split(j, 2*ndim+1, axis=0)[0] for j in fx[1]]
      fx = fx[0]
    else:
      jac = None

    # Split into [f(x), f(x+eps*e_i), f(x-eps*e_i), ...] for basis
    # vectors {e_i}.
    self._fxs = tf.split(fx, 2*ndim+1, axis=0)

    if jac is not None:
      self.f = (self._fxs[0], jac)
    else:
      self.f = self._fxs[0]

    return self.f

  def build_op(self, x):
    """Build operator from a builder f'n for the network 'f' and data 'x'."""
    ndim = x.shape.as_list()[1]
    # d^2/dx_i^2 for each basis vector using finite differences.
    lapl = 0.0
    for i in range(ndim):
      lapl += self._fxs[2*i+1] + self._fxs[2*i+2] - 2*self._fxs[0]
    lapl /= self._eps**2
    self._op = _covariance(self._fxs[0], lapl)
    return self._op

  def __lt__(self, x):
    return False

  def __le__(self, x):
    if isinstance(x, ScaleOperator):
      return self <= x.x
    if isinstance(x, AddOperator):
      return self <= x.x or self <= x.y
    return (isinstance(x, LaplacianOperator) or
            isinstance(x, ExactLaplacianOperator))

  def __gt__(self, x):
    return not self <= x

  def __ge__(self, x):
    return True


def laplacian(f, x):
  """Computes exact Laplacian of f(x). Beware - scales poorly with x."""

  if isinstance(x, list):
    raise ValueError('Input to laplacian must be a single tensor')
  if len(f.shape) == 2:
    return tf.stack(
        [laplacian(f[:, i], x) for i in range(f.shape.as_list()[1])], axis=1)
  elif len(f.shape) == 1:
    dx = tf.reshape(tf.gradients(f, x)[0],
                    (x.get_shape()[0], -1))  # first dim is batch
    ddx = []
    for i in range(dx.get_shape().as_list()[1]):
      ddx.append(tf.reshape(tf.gradients(dx[:, i], x)[0],
                            (x.get_shape()[0], -1))[:, i])
    lapl = tf.add_n(ddx)
    return lapl
  else:
    raise ValueError('Shape of batch must be 1D or 2D')


class ExactLaplacianOperator(LinearOperator):
  """Exact difference Laplacian operator."""

  def __init__(self):
    super(ExactLaplacianOperator, self).__init__()

  def build_op(self, x):
    """Builds operator from a builder f'n for the network 'f' and data 'x'."""

    if isinstance(self.f, tuple):
      f = self.f[0]
    else:
      f = self.f
    lapl = laplacian(f, x)
    self._op = _covariance(f, lapl)
    return self._op

  def __lt__(self, x):
    return False

  def __le__(self, x):
    if isinstance(x, ScaleOperator):
      return self <= x.x
    if isinstance(x, AddOperator):
      return self <= x.x or self <= x.y
    return (isinstance(x, LaplacianOperator) or
            isinstance(x, ExactLaplacianOperator))

  def __gt__(self, x):
    return not self <= x

  def __ge__(self, x):
    return True


class DiagonalOperator(LinearOperator):
  """Operator equivalent to diagonal matrix."""

  def __init__(self, builder):
    super(DiagonalOperator, self).__init__()
    self._builder = builder

  def build_op(self, x):
    kx = self._builder(x)

    if isinstance(self.f, tuple):
      self._op = _covariance(self.f[0], kx * self.f[0])
    else:
      self._op = _covariance(self.f, kx * self.f)

    return self._op


class KernelOperator(LinearOperator):
  """Operator from a symmetric kernel."""

  def __init__(self, kernel):
    super(KernelOperator, self).__init__()
    self._kernel = kernel

  def build_op(self, x):
    x1, x2 = tf.split(x, 2, axis=0)
    fx1, fx2 = tf.split(self.f, 2, axis=0)
    kval = self._kernel(x1, x2)
    self._op = _covariance(fx1, kval * fx2)
    return self._op


class SlownessOperator(LinearOperator):
  """Kernel for slow feature analysis."""

  def build_op(self, x):
    diff = self.f[:-1] - self.f[1:]
    self._op = _covariance(diff, diff)
    return self._op


class SpectralNetwork(object):
  """Class that constructs operators for SpIN and includes training loop."""

  def __init__(self, operator, network, data, params,
               decay=0.0, use_pfor=True, per_example=False):
    """Creates all ops and variables required to train SpIN.

    Args:
      operator: The linear operator to diagonalize.
      network: A function that returns the TensorFlow op for the output of the
        spectral inference network when provided an op for the input.
      data: A TensorFlow op for the input to the spectral inference network.
      params: The trainable parameters of the model built by 'network'.
      decay (optional): The decay parameter for the moving average of the
        network covariance and Jacobian.
      use_pfor (optional): If true, use the parallel_for package to compute
        Jacobians. This is often faster but has higher memory overhead.
      per_example (optional): If true, computes the Jacobian of the network
        output covariance using a more complicated but often faster method.
        This interacts badly with anything that uses custom_gradients, so needs
        to be avoided for some code branches.
    """
    self.operator = operator
    self.data = data
    self.params = params
    self.decay = decay
    self.use_pfor = use_pfor
    self.per_example = per_example

    if per_example and decay != 0.0:
      def network_builder(x):
        """Wraps the function 'network' to compute per-example."""
        def loop_fn(i):
          x_i = tf.expand_dims(tf.gather(x, i), 0)
          features = network(x_i)
          jac = pfor.jacobian(features, params, use_pfor=use_pfor)
          return features, jac

        if use_pfor:
          features, jac = pfor.pfor(loop_fn, x.shape[0])
        else:
          loop_fn_dtypes = [tf.float32, [tf.float32] * len(params)]
          features, jac = pfor.for_loop(loop_fn, loop_fn_dtypes, data.shape[0])
          raise NotImplementedError(
              'use_pfor=False + per_example=True is not yet working.')
        features = _collapse_first_dim(features)
        features.set_shape(network(x).shape)
        jac = [_collapse_first_dim(y) for y in jac]
        for p, j in zip(params, jac):
          j.set_shape(features.shape.as_list() + p.shape.as_list())
        # Note: setting rank=2 so that we use matmul for covariance below
        # instead of batch_matmul.
        return features, jac
    else:
      network_builder = network
    self.network_builder = network_builder

    self.features, self.sigma, self.pi = self._covariances(
        operator, network_builder, data)
    feat_jac = None
    if per_example and decay != 0.0:
      feat_jac = operator.jac
    outputs = self._training_update(
        self.sigma,
        self.pi,
        self.params,
        decay=decay,
        use_pfor=use_pfor,
        features=self.features,
        jac=feat_jac)
    self.loss, self.gradients, self.eigenvalues, self.chol = outputs

  def _moving_average(self, x, c):
    """Creates moving average operation.

    Args:
      x: The tensor or list of tensors of which to take a moving average.
      c: The decay constant of the moving average, between 0 and 1.
        0.0 = the moving average is constant
        1.0 = the moving averge has no memory

    Returns:
      ma: Moving average variables.
      ma_update: Op to update moving average.
    """
    if isinstance(x, list):
      mas = [self._moving_average(y, c) for y in x]
      return [m[0] for m in mas], [m[1] for m in mas]
    if len(x.shape) == 2 and x.shape[0] == x.shape[1]:
      ma = tf.Variable(tf.eye(x.shape.as_list()[0]), trainable=False)
    else:
      ma = tf.Variable(tf.zeros_like(x), trainable=False)
    ma_update = tf.assign(ma, (1-c)*ma + c*tf.reshape(x, ma.shape))
    return ma, ma_update

  def _covariances(self, operator, network, x):
    """Constructs loss with custom gradient for SpIN.

    Args:
      operator: The linear operator to diagonalize.
      network: A function that returns the TensorFlow op for the output of the
        spectral inference network when provided an op for the input.
      x: The data used as input to network.

    Returns:
      u: The output of the spectral inference network.
      sigma: The covariance of the outputs of the network.
      pi: The matrix of network output covariances multiplied by the linear
        operator to diagonalize. See paper for explicit definition.
    """
    u, pi = operator.build(network, x)
    sigma = _covariance(u, u)
    sigma.set_shape((u.shape[1], u.shape[1]))
    pi.set_shape((u.shape[1], u.shape[1]))
    return u, sigma, pi

  def _training_update(self,
                       sigma,
                       pi,
                       params,
                       decay=0.0,
                       use_pfor=False,
                       features=None,
                       jac=None):
    """Makes gradient and moving averages.

    Args:
      sigma: The covariance of the outputs of the network.
      pi: The matrix of network output covariances multiplied by the linear
        operator to diagonalize. See paper for explicit definition.
      params: The trainable parameters.
      decay (optional): The decay parameter for the moving average of the
        network covariance and Jacobian.
      use_pfor (optional): If true, use the parallel_for package to compute
        Jacobians. This is often faster but has higher memory overhead.
      features (optional): The output features of the spectral inference
        network. Only necessary if per_example=True.
      jac (optional): The Jacobian of the network. Only necessary if
        per_example=True.

    Returns:
      loss: The loss function for SpIN - the sum of eigenvalues.
      gradients: The approximate gradient of the loss using moving averages.
      eigvals: The full array of eigenvalues, rather than just their sum.
      chol: The Cholesky decomposition of the covariance of the network outputs,
        which is needed to demix the network outputs.
    """
    if isinstance(decay, float):
      assert decay >= 0.0 and decay < 1.0
    if decay == 0.0:
      # Equivalent to not using the moving averages at all.
      loss, eigval, chol = _objective(sigma, pi)  # pylint: disable=unbalanced-tuple-unpacking
      gradients = tf.gradients(loss, params)
    else:
      if jac is not None:
        sig_feat_jac = pfor.jacobian(sigma, features, use_pfor=use_pfor)
        sigma_jac = [tf.tensordot(sig_feat_jac, y, axes=2) for y in jac]
      else:
        sigma_jac = pfor.jacobian(sigma, params, use_pfor=use_pfor)
      for p, sj in zip(params, sigma_jac):
        sj.set_shape(sigma.shape.as_list() + p.shape.as_list())

      sigma_avg, update_sigma = self._moving_average(sigma, decay)
      sigma_jac_avg, update_sigma_jac = self._moving_average(sigma_jac, decay)
      n = tf.reduce_prod(tf.shape(sigma))
      with tf.control_dependencies(update_sigma_jac + [update_sigma]):
        loss, eigval, chol = _objective(sigma_avg, pi)  # pylint: disable=unbalanced-tuple-unpacking
        sigma_back = tf.gradients(loss, sigma_avg)[0]

        gradients = []
        for s, p, g in zip(sigma_jac_avg, params, tf.gradients(loss, params)):
          gradients.append(
              tf.reshape(
                  tf.matmul(
                      tf.reshape(sigma_back,
                                 (1, n)), tf.reshape(s, (n, -1))), p.shape) + g)

    return loss, gradients, eigval, chol

  def train(
      self,
      optim,
      iterations,
      logging_config,
      stats_hooks,
      plotting_hooks=None,
      show_plots=False,
      global_step=None,
      data_for_plotting=None):
    """Training loop for SpIN, with hooks for logging and plotting.

    Args:
      optim: The TensorFlow optimizer to minimize the SpIN loss.
      iterations: The number of iterations to train for.
      logging_config: A dictionary for logging. The field 'config' is logged
        at the beginning of training, with metadata about the run, while the
        fields 'saver_path' and 'saver_name' are for setting up checkpointing
        and 'log_image_every' and 'save_params_every' set the number of
        iterations after which logging and checkpoint saving occur.
      stats_hooks: A dictionary with two fields, 'create' and 'update', both of
        which are functions that take no arguments. 'create' sets up the data
        structures for logging stats while 'update' updates them.
      plotting_hooks (optional): If show_plots is true, this dictionary must be
        provided. Has the same format as 'stats_hooks'.
      show_plots (optional): A boolean. If true, will plot results to the GUI.
      global_step (optional): A TensorFlow op that tracks the number of
        iterations. If none is provided, one is created.
      data_for_plotting (optional): If different data is needed for updating
        plots than for training, this op will return that data.

    Returns:
      A dictionary of statistics accumulated during the training run.
    """

    if show_plots:
      plt.ion()
      if plotting_hooks is None:
        raise ValueError('Plotting hooks are required if show_plots=True')
      plots = plotting_hooks['create']()

    saver_path = logging_config['saver_path']
    saver = tf.train.Saver(var_list=self.params)

    if global_step is None:
      global_step = tf.Variable(0.0, trainable=False, name='global_step')

    if data_for_plotting is not None:
      features_for_plotting = self.network_builder(data_for_plotting)
      # If per_example is true, the network builder will return a
      # (features, jacobian) tuple. For plotting, we can discard the latter.
      if isinstance(features_for_plotting, tuple):
        features_for_plotting = features_for_plotting[0]
      features_for_plotting = tf.transpose(
          tf.matrix_triangular_solve(self.chol,
                                     tf.transpose(features_for_plotting)))

    neig = self.features.shape.as_list()[-1]
    stats = stats_hooks['create'](iterations, neig)

    logging.info(logging_config['config'])

    update_global_step = tf.assign(global_step, global_step + 1)
    step = optim.apply_gradients(zip(self.gradients, self.params))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print('Initialized variables')
      for t in range(iterations):
        sess.run(update_global_step)
        loss_, eigenvalues_, _ = sess.run([self.loss, self.eigenvalues, step])
        stats_hooks['update'](t, loss_, eigenvalues_, **stats)
        current_stats = dict((key, stats[key][t]) for key in stats)

        logging.info(current_stats)

        if t % logging_config['save_params_every'] == 0:
          saver.save(sess,
                     saver_path + '/' + logging_config['saver_name'],
                     global_step=t)

        if t % logging_config['log_image_every'] == 0:
          if data_for_plotting is not None and show_plots:
            outputs = sess.run(features_for_plotting)
            inputs = sess.run(data_for_plotting)
            plotting_hooks['update'](t, outputs, inputs, *plots, **stats)

            plt.show()
            plt.pause(0.01)

    return stats
