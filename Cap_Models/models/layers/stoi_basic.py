
import numpy as np
import six
import sonnet as snt
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nest


class WeightedAttentionPooling(snt.AbstractModule):


    def __init__(self, gate_nn, message_nn):

        super(WeightedAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = tf.random_normal([1])

    def _build(self, x, index, weights):

        gate = self.gate_nn(x)

        gate = gate - tf.gather(tf.segment_max(gate, index),index, axis = 0)
        gate = (weights ** self.pow) * tf.exp(gate)
        # gate = weights * gate.exp()
        # gate = gate.exp()

        a = tf.segment_sum(gate, index)
        b = (tf.gather(a, index, axis = 0) + 1e-10)
        gate = gate / b

        x = self.message_nn(x)
        c = gate * x
        out = tf.segment_sum(c, index)

        return out

    def _init(self, key):
        if self.initializers:
            return self.initializers.get(key, None)


class SimpleNetwork(snt.AbstractModule):

    def __init__(
        self, input_dim, output_dim, hidden_layer_dims, batchnorm=False):
        super(SimpleNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.batchnorm = batchnorm

    def _build(self, x):

        self.dims = [self.input_dim] + self.hidden_layer_dims

        for i in range(len(self.dims) - 1):
            h_1 = snt.Linear(self.dims[i], use_bias = True)(x)
            x = tf.nn.selu(h_1)

        out = snt.Linear(self.output_dim, use_bias=True)(x)

        return out

    def _init(self, key):
        if self.initializers:
            return self.initializers.get(key, None)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size = 128,
                     initializer_range = 0.02,
                     word_embedding_name = "word_embeddings",
                     use_one_hot_embeddings = False):


  embedding_table = tf.get_variable(
      name = word_embedding_name,
      shape = [vocab_size, embedding_size],
      initializer = create_initializer(initializer_range))

  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:

    output = tf.matmul(input_ids, embedding_table)
  else:
    output = tf.gather(embedding_table, flat_input_ids)

  input_shape = get_shape_list(input_ids)

  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type = False,
                            token_type_ids = None,
                            token_type_vocab_size = 16,
                            token_type_embedding_name = "token_type_embeddings",
                            use_position_embeddings = False,
                            position_embedding_name = "position_embeddings",
                            initializer_range = 0.02,
                            max_position_embeddings = 512,
                            dropout_prob=0.1):

  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))

    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):

  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

def dropout(input_tensor, dropout_prob):


  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def get_shape_list(tensor, expected_rank=None, name=None):

  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):

  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

