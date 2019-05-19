# Spectral Inference Networks (SpIN)
This package provides an implementation of Spectral Inference Networks,
as in [Pfau, Petersen, Agarwal, Barrett and Stachenfeld (2019)](https://arxiv.org/abs/1806.02215).

This is not an officially supported Google product.

## Prerequisites
SpIN requires a working installation of Python and TensorFlow. We recommend
running it on GPU for faster convergence.

If you want to make use of the GUI (on by default) you will also need Tcl/Tk
installed on your system.

## Installation
After cloning the repo, run pip to install the package and its Python
dependencies:

```bash
cd spectral_inference_networks
pip install .
```

## Usage
Training a spectral inference network is similar to most other deep learning
pipelines: you must construct a data source, network architecture and optimizer.
What makes spectral inference networks unique is that instead of a loss you
provide a linear operator to diagonalize. The code expects an object of the
LinearOperator class, which can be constructed from a similarity kernel or by
other means. LinearOperator objects can be added together or multiplied by a
scalar.

Below is a minimal example of training spectral inference networks:

```python
import tensorflow as tf
import spectral_inference_networks as spin

batch_size = 1024
input_dim = 10
num_eigenvalues = 5
iterations = 1000  # number of training iterations

# Create variables for simple MLP
w1 = tf.Variable(tf.random.normal([input_dim, 64]))
w2 = tf.Variable(tf.random.normal([64, num_eigenvalues]))

b1 = tf.Variable(tf.random.normal([64]))
b2 = tf.Variable(tf.random.normal([num_eigenvalues]))

# Create function to construct simple MLP
def network(x):
  h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
  return tf.matmul(h1, w2) + b2

data = tf.random.normal([batch_size, input_dim])  # replace with actual data
# Squared exponential kernel.
kernel = lambda x, y: tf.exp(-(tf.norm(x-y, axis=1, keepdims=True)**2))
linop = spin.KernelOperator(kernel)
optim = tf.train.AdamOptimizer()

# Constructs the internal training ops for spectral inference networks.
spectral_net = spin.SpectralNetwork(
    linop,
    network,
    data,
    [w1, w2, b1, b2])

# Trivial defaults for logging and stats hooks.
logging_config = {
    'config': {},
    'log_image_every': iterations,
    'save_params_every': iterations,
    'saver_path': '/tmp',
    'saver_name': 'example',
}

stats_hooks = {
    'create': spin.util.create_default_stats,
    'update': spin.util.update_default_stats,
}

# Executes the training of spectral inference networks.
stats = spectral_net.train(
    optim,
    iterations,
    logging_config,
    stats_hooks)
```

We provide two examples in the `examples` folder, which you can run as follows:

```bash
python spectral_inference_networks/examples/hydrogen.py
```

and:

```bash
python spectral_inference_networks/examples/atari.py
```
These correspond to experiments in section 5.1 and C.3 of the paper.
Each example comes with a range of supported command line arguments. Please
take a look in the source code for each example for further information and have
a play with the many options.

## Giving Credit
If you use this code in your work, we ask that you cite the paper:

David Pfau, Stig Petersen, Ashish Agarwal, David Barrett, Kim Stachenfeld.
"Spectral Inference Networks: Unifying Deep and Spectral Learning."
_The 7th International Conference on Learning Representations (ICLR)_ (2019).

## Acknowledgements
Special thanks to James Spencer for help with the open-source implementation of
the code.
