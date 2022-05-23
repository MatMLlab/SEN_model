"""
Capsule layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from monty.collections import AttrDict
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf
from tensorflow.python.util import nest
import tensorflow_probability as tfp

import geo_block
from models.layers.batch_mlp import BatchMLP

tfd = tfp.distributions


class CapsuleLayer(snt.AbstractModule):
  """Implementation of a capsule layer."""

  _n_transform_params = 6

  def __init__(self, n_caps, n_caps_dims, n_votes, n_caps_params=None,
               n_hiddens=128, learn_vote_scale=False, deformations=True,
               noise_type=None, noise_scale=0., similarity_transform=True,
               caps_dropout_rate=0.0):
    """Builds the module.
    """
    super(CapsuleLayer, self).__init__()
    self._n_caps = n_caps
    self._n_caps_dims = n_caps_dims
    self._n_caps_params = n_caps_params

    self._n_votes = n_votes
    self._n_hiddens = nest.flatten(n_hiddens)
    self._learn_vote_scale = learn_vote_scale
    self._deformations = deformations
    self._noise_type = noise_type
    self._noise_scale = noise_scale

    self._similarity_transform = similarity_transform
    self._caps_dropout_rate = caps_dropout_rate

    assert n_caps_dims == 2, ('This is the only value implemented now due to '
                              'the restriction of similarity transform.')

  def _build(self, features, parent_transform=None, parent_presence=None):

    batch_size = tf.shape(features)[0]
    batch_shape = [batch_size, self._n_caps]

    # Predict capsule and additional params from the input encoding.
    # [B, n_caps, n_caps_dims]
    if self._n_caps_params is not None:

      # Use separate parameters to do predictions for different capsules.
      mlp = BatchMLP(self._n_hiddens + [self._n_caps_params])
      raw_caps_params = mlp(features)

      caps_params = tf.reshape(raw_caps_params,
                               batch_shape + [self._n_caps_params])

    else:
      assert features.shape[:2].as_list() == batch_shape
      caps_params = features

    if self._caps_dropout_rate == 0.0:
      caps_exist = tf.ones(batch_shape + [1], dtype=tf.float32)
    else:
      pmf = tfd.Bernoulli(1. - self._caps_dropout_rate, dtype=tf.float32)
      caps_exist = pmf.sample(batch_shape + [1])

    caps_params = tf.concat([caps_params, caps_exist], -1)

    output_shapes = (
        [self._n_votes, self._n_transform_params],  # CPR_dynamic
        [1, self._n_transform_params],  # CCR
        [1],  # per-capsule presence
        [self._n_votes],  # per-vote-presence
        [self._n_votes],  # per-vote scale
    )

    splits = [np.prod(i).astype(np.int32) for i in output_shapes]
    n_outputs = sum(splits)

    caps_mlp = BatchMLP([self._n_hiddens, n_outputs], use_bias=False)
    all_params = caps_mlp(caps_params)
    all_params = tf.split(all_params, splits, -1)
    res = [tf.reshape(i, batch_shape + s)
           for (i, s) in zip(all_params, output_shapes)]

    cpr_dynamic = res[0] # OP static

    # add bias to all remaining outputs
    res = [snt.AddBias()(i) for i in res[1:]]

    ccr, pres_logit_per_caps, pres_logit_per_vote, scale_per_vote = res

    if self._caps_dropout_rate != 0.0:
      pres_logit_per_caps += geo_block.safe_log(caps_exist)

    cpr_static = tf.get_variable('cpr_static', shape=[1, self._n_caps,
                                                      self._n_votes,
                                                      self._n_transform_params])

    def add_noise(tensor):
      """Adds noise to tensors."""
      if self._noise_type == 'uniform':
        noise = tf.random.uniform(tf.shape(tensor),
                                  minval=-.5, maxval=.5) * self._noise_scale

      elif self._noise_type == 'logistic':
        pdf = tfd.Logistic(0., self._noise_scale)
        noise = pdf.sample(tensor.shape)

      elif not self._noise_type:
        noise = 0.

      else:
        raise ValueError('Invalid noise type: "{}".'.format(self._noise_type))

      return tensor + noise

    pres_logit_per_caps = add_noise(pres_logit_per_caps) # a_k of object capsule
    pres_logit_per_vote = add_noise(pres_logit_per_vote) # a_kn of object capsule

    # this is for hierarchical
    if parent_transform is None:
      ccr = self._make_transform(ccr)
    else:
      ccr = parent_transform

    if not self._deformations:
      cpr_dynamic = tf.zeros_like(cpr_dynamic)

    cpr = self._make_transform(cpr_dynamic + cpr_static)

    ccr_per_vote = snt.TileByDim([2], [self._n_votes])(ccr)
    votes = tf.matmul(ccr_per_vote, cpr) # v_kn = OV_k * OP_kn

    if parent_presence is not None:
      pres_per_caps = parent_presence
    else:
      pres_per_caps = tf.nn.sigmoid(pres_logit_per_caps) # a_k of object capsule

    pres_per_vote = pres_per_caps * tf.nn.sigmoid(pres_logit_per_vote) # a_k * a_kn

    if self._learn_vote_scale:
      # for numerical stability
      scale_per_vote = tf.nn.softplus(scale_per_vote + .5) + 1e-2
    else:
      scale_per_vote = tf.zeros_like(scale_per_vote) + 1.

    abc = tf.nn.l2_loss(cpr_dynamic)
    dynamic_weights_l2 = abc  / tf.cast(batch_size,tf.float32)
    return AttrDict(
        vote = votes, # v_kn = OV_k * OP_kn
        scale = scale_per_vote, # c_k
        vote_presence = pres_per_vote, # a_k * a_kn
        pres_logit_per_caps = pres_logit_per_caps, # a_k of object capsule
        pres_logit_per_vote = pres_logit_per_vote, # a_kn of object capsule
        dynamic_weights_l2 = dynamic_weights_l2,
        raw_caps_params = raw_caps_params,
        raw_caps_features = features, # output of ocae encoder
    )

  def _make_transform(self, params):
    return geo_block.geometric_transform(params, self._similarity_transform,
                                         nonlinear=True, as_matrix=True)


class OrderInvariantCapsuleLikelihood(snt.AbstractModule):
  """Permutation-invariant capsule layer used in constellation experiments."""
  #CCAE

  OutputTuple = collections.namedtuple('CapsuleLikelihoodTuple',  # pylint:disable=invalid-name
                                       ('prediction posterior_pre log_prob vote_presence winner '
                                        'winner_pres is_from_capsule '
                                        'mixing_logits mixing_log_prob '
                                        'soft_winner soft_winner_pres '
                                        'posterior_mixing_probs'))

  def __init__(self, n_votes, votes, scales, vote_presence_prob, raw_caps_params, pdf='normal'):
    super(OrderInvariantCapsuleLikelihood, self).__init__()
    self._n_votes = n_votes
    self._votes = votes
    self._scales = scales
    self._vote_presence_prob = vote_presence_prob
    self._raw_caps_params = raw_caps_params
    self._pdf = pdf

  def _build(self, x, pres_logit_per_vote, pres_logit_per_caps, presence=None):

    batch_size, n_input_points = x.shape[:2].as_list()

    expanded_votes = tf.expand_dims(self._votes, 1)
    expanded_scale = tf.expand_dims(tf.expand_dims(self._scales, 1), -1)
    vote_component_pdf = self._get_pdf(expanded_votes, expanded_scale)

    # [B, n_points, n_caps, n_votes, n_input_dims]
    expanded_x = tf.expand_dims(x, 2)
    vote_log_prob_per_dim = vote_component_pdf.log_prob(expanded_x)
    # [B, n_points, n_votes]
    vote_log_prob = tf.reduce_sum(vote_log_prob_per_dim, -1)
    dummy_vote_log_prob = tf.zeros([batch_size, n_input_points, 1])
    dummy_vote_log_prob -= 2. * tf.log(10.)
    vote_log_prob = tf.concat([vote_log_prob, dummy_vote_log_prob], 2)

    # [B, n_points, n_votes]
    mixing_logits = geo_block.safe_log(self._vote_presence_prob)

    dummy_logit = tf.zeros([batch_size, 1]) -2. * tf.log(10.)
    mixing_logits = tf.concat([mixing_logits, dummy_logit], 1)

    mixing_log_prob = mixing_logits - tf.reduce_logsumexp(mixing_logits, 1,
                                                          keepdims=True)

    expanded_mixing_logits = tf.expand_dims(mixing_log_prob, 1)
    mixture_log_prob_per_component\
      = tf.reduce_logsumexp(expanded_mixing_logits + vote_log_prob, 2)

    if presence is not None:
      presence = tf.to_float(presence)
      mixture_log_prob_per_component *= presence

    mixture_log_prob_per_example\
      = tf.reduce_sum(mixture_log_prob_per_component, 1)

    mixture_log_prob_per_batch = tf.reduce_mean(mixture_log_prob_per_example)

    # [B, n_points, n_votes]
    posterior_mixing_logits_per_point = expanded_mixing_logits + vote_log_prob
    # [B, n_points]
    winning_vote_idx = tf.argmax(
        posterior_mixing_logits_per_point[:, :, :-1], 2)

    batch_idx = tf.expand_dims(tf.range(batch_size, dtype=tf.int64), -1)
    batch_idx = snt.TileByDim([1], [winning_vote_idx.shape[-1]])(batch_idx)

    idx = tf.stack([batch_idx, winning_vote_idx], -1)
    winning_vote = tf.gather_nd(self._votes, idx)
    winning_pres = tf.gather_nd(self._vote_presence_prob, idx)
    vote_presence = tf.greater(mixing_logits[:, :-1],
                               mixing_logits[:, -1:])

    # the first four votes belong to the square
    is_from_capsule = winning_vote_idx // self._n_votes

    posterior_mixing_probs = tf.nn.softmax(
        posterior_mixing_logits_per_point, -1)[Ellipsis, :-1]

    assert winning_vote.shape == x.shape

    return self.OutputTuple(
        log_prob=mixture_log_prob_per_batch,
        vote_presence=tf.to_float(vote_presence),
        winner=winning_vote,
        winner_pres=winning_pres,
        is_from_capsule=is_from_capsule,
        mixing_logits=mixing_logits,
        mixing_log_prob=mixing_log_prob,
        # TODO(adamrk): this is broken
        soft_winner=tf.zeros_like(winning_vote),
        soft_winner_pres=tf.zeros_like(winning_pres),
        posterior_mixing_probs=posterior_mixing_probs,
    )

  def _get_pdf(self, votes, scale):

    if self._pdf == 'normal':
      pdf = tfd.Normal(votes, scale)

    elif self._pdf == 'student':
      df = tf.get_variable('df', initializer=2., dtype=tf.float32)
      df = tf.nn.softplus(df) + 2.
      pdf = tfd.StudentT(df, votes, scale)

    else:
      raise ValueError('Distribution "{}" not '
                       'implemented!'.format(self._pdf))

    return pdf

  def connect(self, x, presence=None):
    return self.__call__(x, presence)

  def log_prob(self, x, presence=None):
    return self.connect(x, presence).log_prob

  def explain(self, x, presence=None):
    o = self.connect(x, presence)
    return o.winner


class CapsuleLikelihood(OrderInvariantCapsuleLikelihood):
  """Capsule voting mechanism."""

  def __init__(self, votes, scales, vote_presence_prob, raw_caps_params, pdf='normal'):
    super(CapsuleLikelihood, self).__init__(1, votes, scales,
                                            vote_presence_prob, raw_caps_params, pdf)
    self._n_caps = int(self._votes.shape[1])

  def _build(self, x, pres_logit_per_vote, pres_logit_per_caps, presence=None):

    # x is [B, n_input_points, n_input_dims]
    batch_size, n_input_points = x.shape[:2].as_list()

    # votes and scale have shape [B, n_caps, n_input_points, n_input_dims|1]
    # since scale is a per-caps scalar and we have one vote per capsule
    # Gaussian Mixture build from ov and op
    vote_component_pdf = self._get_pdf(self._votes,
                                       tf.expand_dims(self._scales, -1))

    # expand along caps dimensions -> [B, 1, n_input_points, n_input_dims]
    expanded_x = tf.expand_dims(x, 1)
    vote_log_prob_per_dim = vote_component_pdf.log_prob(expanded_x)
    # [B, n_caps, n_input_points]
    vote_log_prob = tf.reduce_sum(vote_log_prob_per_dim, -1) # log gaussian pose

    # prediction output
    #raw_caps = tf.nn.selu(snt.BatchApply(snt.Linear(64, use_bias = True))(self._raw_caps_params))
    #Tansformer = QKVAttention()
    #raw_caps = Tansformer(raw_caps, raw_caps, raw_caps)
    raw_caps = tf.nn.selu(snt.BatchApply(snt.Linear(16, use_bias = True))(self._raw_caps_params))
    scale_with_presence = tf.multiply(raw_caps, pres_logit_per_vote)
    scale_with_capsule = tf.reduce_mean(scale_with_presence, axis=-1)
    pres_logit_per_caps = tf.reshape(pres_logit_per_caps, (-1, 10))
    cap_scale_with_pres = tf.multiply(scale_with_capsule, pres_logit_per_caps)
    prediction = tf.nn.relu(snt.Linear(1)(cap_scale_with_pres))

    bs = tf.shape(x)[0]
    #dummy_vote_log_prob = tf.zeros([batch_size, 1, n_input_points])
    dummy_vote_log_prob = tf.zeros([bs, 1, n_input_points])
    dummy_vote_log_prob -= 2. * tf.log(10.)

    # [B, n_caps + 1, n_input_points]
    vote_log_prob = tf.concat([vote_log_prob, dummy_vote_log_prob], 1) # log gaussian

    # [B, n_caps, n_input_points]
    mixing_logits = geo_block.safe_log(self._vote_presence_prob) # log(a_k * a_kn)

    dummy_logit = tf.zeros([bs, 1, 1]) -2. * tf.log(10.)
    dummy_logit = snt.TileByDim([2], [n_input_points])(dummy_logit)

    mixing_logits = tf.concat([mixing_logits, dummy_logit], 1) # a_k * a_kn
    mixing_log_prob = mixing_logits - tf.reduce_logsumexp(mixing_logits, 1,
                                                          keepdims=True)

    mixture_log_prob_per_point = tf.reduce_logsumexp(
        mixing_logits + vote_log_prob, 1) # before log

    if presence is not None:
      presence = tf.cast(presence, tf.float32)
      mixture_log_prob_per_point *= presence

    mixture_log_prob_per_example = tf.reduce_sum(mixture_log_prob_per_point, 1)

    mixture_log_prob_per_batch = tf.reduce_mean(mixture_log_prob_per_example) # part log likelihood

    posterior_mixing_logits_per_point = mixing_logits + vote_log_prob # a_k * a_kn * Gaussian, a_kn_posterior

    winning_vote_idx = tf.argmax(
        posterior_mixing_logits_per_point[:, :-1], 1) # mixture component

    batch_idx = tf.expand_dims(tf.range(tf.cast(bs, dtype = 'int64'), dtype=tf.int64), 1)
    batch_idx = snt.TileByDim([1], [n_input_points])(batch_idx)

    point_idx = tf.expand_dims(tf.range(n_input_points, dtype=tf.int64), 0)
    point_idx = snt.TileByDim([0], [bs])(point_idx)

    idx = tf.stack([batch_idx, winning_vote_idx, point_idx], -1)
    winning_vote = tf.gather_nd(self._votes, idx)
    winning_pres = tf.gather_nd(self._vote_presence_prob, idx)
    vote_presence = tf.greater(mixing_logits[:, :-1],
                               mixing_logits[:, -1:])

    is_from_capsule = winning_vote_idx // self._n_votes

    posterior_mixing_probs = tf.nn.softmax(posterior_mixing_logits_per_point, 1)

    dummy_vote = tf.get_variable('dummy_vote', shape = [1, 1, 16, 6])
    #dummy_vote = tf.get_variable('dummy_vote', shape=self._votes[:1, :1].shape)
    dummy_vote = snt.TileByDim([0], [bs])(dummy_vote)
    dummy_pres = tf.zeros([bs, 1, n_input_points])

    votes = tf.concat((self._votes, dummy_vote), 1) # v_kn, mu_kn ?
    pres = tf.concat([self._vote_presence_prob, dummy_pres], 1) # a_k * a_kn

    soft_winner = tf.reduce_sum(
        tf.expand_dims(posterior_mixing_probs, -1) * votes, 1)
    soft_winner_pres = tf.reduce_sum(
        posterior_mixing_probs * pres, 1)

    posterior_mixing_probs = tf.transpose(
        posterior_mixing_probs[:, :-1], (0, 2, 1))


    return self.OutputTuple(prediction = prediction,
                            posterior_pre = mixture_log_prob_per_example,
                            log_prob = mixture_log_prob_per_batch, # part log likelihood
                            vote_presence = tf.cast(vote_presence, tf.float32), # log(a_k * a_kn)
                            winner = winning_vote, # v_kn
                            winner_pres = winning_pres, # a_k * a_kn
                            soft_winner = soft_winner,
                            soft_winner_pres = soft_winner_pres,
                            posterior_mixing_probs = posterior_mixing_probs,
                            is_from_capsule = is_from_capsule,
                            mixing_logits = mixing_logits,
                            mixing_log_prob = mixing_log_prob
                            )


def _capsule_entropy(caps_presence_prob, k=1, **unused_kwargs):
  """Computes entropy in capsule activations."""
  del unused_kwargs

  within_example = geo_block.normalize(caps_presence_prob, 1)
  within_example = geo_block.safe_ce(within_example, within_example * k)

  between_example = tf.reduce_sum(caps_presence_prob, 0)
  between_example = geo_block.normalize(between_example, 0)
  between_example = geo_block.safe_ce(between_example, between_example * k)

  return within_example, between_example


# kl(aggregated_prob||uniform)
def _neg_capsule_kl(caps_presence_prob, **unused_kwargs):
  del unused_kwargs

  num_caps = int(caps_presence_prob.shape[-1])
  return _capsule_entropy(caps_presence_prob, k=num_caps)


# l2(aggregated_prob - constant)
def _caps_pres_l2(caps_presence_prob, num_classes=10.,
                  within_example_constant=0., **unused_kwargs):
  """Computes l2 penalty on capsule activations."""

  del unused_kwargs

  batch_size, num_caps = caps_presence_prob.shape.as_list()

  if within_example_constant == 0.:
    within_example_constant = float(num_caps) / num_classes

  between_example_constant = float(batch_size) / num_classes

  within_example = tf.nn.l2_loss(
      tf.reduce_sum(caps_presence_prob, 1)
      - within_example_constant) / batch_size * 2.

  between_example = tf.nn.l2_loss(
      tf.reduce_sum(caps_presence_prob, 0)
      - between_example_constant) / num_caps * 2.

  # neg between example because it's subtracted from the loss later on
  return within_example, -between_example


def sparsity_loss(loss_type, *args, **kwargs):
  """Computes capsule sparsity loss according to the specified type."""

  if loss_type == 'entropy':
    sparsity_func = _capsule_entropy

  elif loss_type == 'kl':
    sparsity_func = _neg_capsule_kl

  elif loss_type == 'l2':
    sparsity_func = _caps_pres_l2

  else:
    raise ValueError(
        'Invalid sparsity loss: "{}"'.format(loss_type))

  return sparsity_func(*args, **kwargs)
