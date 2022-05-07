
import functools
from monty.collections import AttrDict
import sonnet as snt
import tensorflow.compat.v1 as tf


def squash(vectors, axis=-1):
  """
  The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
  :param vectors: some vectors to be squashed, N-dim tensor
  :param axis: the axis to squash
  :return: a Tensor with same shape as input vectors
  """
  s_squared_norm = tf.reduce_sum(tf.square(vectors), axis = 1, keepdims=True)
  scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-07)
  return scale * vectors


class _Memoize(object):
  """Class implementing a lookup table for memoization."""

  def __init__(self, arg_transform=None):
    super(_Memoize, self).__init__()
    self.memo = {}
    self.arg_transform = arg_transform

  def __call__(self, func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      """Wrapper implementing lookup tables."""

      hashed_args = args, kwargs
      if self.arg_transform is not None:
        hashed_args = self.arg_transform(*hashed_args)

      hashed_args = tuple(hashed_args[0]) + tuple(hashed_args[1].items())
      if hashed_args not in self.memo:
        self.memo[hashed_args] = func(*args, **kwargs)

      return self.memo[hashed_args]

    return wrapper


memoize_func = _Memoize  # pylint:disable=invalid-name


def memoize_arg_transform(args, kwargs):
  """Transform unhashable types to their string representation."""
  # TODO(adamrk): this is rather unsafe.
  def transform(x):
    try:
      hash(x)
    except TypeError:
      x = str(x)

    return x

  args = (transform(arg) for arg in args)
  kwargs = {transform(k): transform(v) for (k, v) in kwargs.items()}
  return args, kwargs

Memoize = functools.partial(memoize_func,  # pylint:disable=invalid-name
                            arg_transform=memoize_arg_transform)


class Model(snt.AbstractModule):
  """Generic model class."""

  def _loss(self, data, results):
    """Defines the loss."""
    return NotImplemented

  def _report(self, data, results):
    """Defines any values that should be logged/reported."""
    reports = {k: v for k, v in results.items()
               if isinstance(v, tf.Tensor) and v.shape == tuple()}
    return AttrDict(reports)

  @Memoize()
  def connect(self, data, *args, **kwargs):

    # try to convert a namedtuple to a dict
    try:
      data = data._asdict()
    except AttributeError:
      pass

    return self._do_call(data, *args, **kwargs)

  def _do_call(self, data, *args, **kwargs):
    return self.__call__(data, *args, **kwargs)

  @Memoize()
  def make_target(self, data, *unused_args, **unused_kwargs):
    del unused_args
    del unused_kwargs

    res = self.connect(data)
    loss = self._loss(data, res)

    return loss, None

  @Memoize()
  def make_report(self, data):

    res = self.connect(data)
    exprs = AttrDict(loss=self._loss(data, res))
    exprs.update(self._report(data, res))

    for k, v in exprs.items():
      if not isinstance(v, tf.Tensor):
        exprs[k] = tf.convert_to_tensor(v)

    return exprs

  @Memoize()
  def make_plot(self, data, name=None):
    res = self.connect(data)
    plots = self._plot(data, res)
    plots = [_append_name(p, name) for p in plots]
    return plots


def _append_name(data_dict, name):
  if name is None:
    return data_dict

  return {'{}_{}'.format(name, k): v for (k, v) in data_dict.items()}
