
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nest


class BatchLinear(snt.AbstractModule):
  """Performs k independent linear transformations of k vectors."""

  def __init__(self, n_units, tile_dims=(0,), use_bias=True, initializers=None):

    super(BatchLinear, self).__init__()
    self._n_units = n_units
    self._tile_dims = sorted(tile_dims)
    self._use_bias = use_bias
    self.initializers = snt.python.modules.util.check_initializers(
        initializers, {'w', 'b'} if use_bias else {'w'})

  def _build(self, x):


    # batch_size, n_inputs, n_dims = x.shape.as_list()
    shape = x.shape.as_list()
    #shape = tf.shape(x)

    if 'w' not in self.initializers:
      stddev = 1 / math.sqrt(shape[-1])
      self.initializers['w'] = tf.truncated_normal_initializer(
          stddev=stddev)

    weights_shape = shape + [self._n_units]
    tiles = []
    for i in self._tile_dims:
      tiles.append(weights_shape[i])
      weights_shape[i] = 1

    weights = tf.get_variable('weights', shape=weights_shape,
                              initializer=self._init('w'))

    tiles_1 = [tf.shape(x)[0]]
    weights = snt.TileByDim(self._tile_dims, tiles_1)(weights)

    x = tf.expand_dims(x, -2)
    y = tf.matmul(x, weights)
    y = tf.squeeze(y, -2)

    if self._use_bias:
      if 'b' not in self.initializers:
        self.initializers['b'] = tf.zeros_initializer()

      init = dict(b=self._init('b'))
      bias_dims = [i for i in range(len(shape)) if i not in self._tile_dims]
      add_bias = snt.AddBias(bias_dims=bias_dims, initializers=init)
      y = add_bias(y)

    return y

  def _init(self, key):
    if self.initializers:
      return self.initializers.get(key, None)


class BatchMLP(snt.AbstractModule):

  def __init__(self, n_hiddens,
               activation = tf.nn.selu,
               activation_final = tf.nn.selu,
               activate_final = False,
               initializers = None,
               use_bias = True,
               tile_dims = (0,)):

    super(BatchMLP, self).__init__()
    self._n_hiddens = nest.flatten(n_hiddens)
    self._activation = activation
    self._activation_final = activation_final
    self._activate_final = activate_final
    self._initializers = initializers
    self._use_bias = use_bias
    self._tile_dims = tile_dims

  def _build(self, x):

    h = x
    for n_hidden in self._n_hiddens[:-1]:
      layer = BatchLinear(n_hidden, initializers=self._initializers,
                          use_bias=True)
      h = self._activation(layer(h))

    layer = BatchLinear(self._n_hiddens[-1], initializers=self._initializers,
                        use_bias=self._use_bias)
    h = layer(h)
    if self._activate_final:
      h = self._activation_final(h)

    return h