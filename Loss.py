"""
Loss for material capsule
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
import numpy as np
from scipy import optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sonnet as snt

class Pre_loss(object):
    def __init__(self, k=2):
        self.k = k
    def mse(self, y_pred):
        mse = y_pred.best_pre_loss

        return mse

# calculate the mse loss between prediction and real bandgap
def pred_loss(prediction, labels, n_classes = 1):
  """Classification probe with stopped gradient on features."""

  def _pre_probe(features):
      logits = snt.Linear(1)(features)
      logits_1 = tf.reduce_mean(features, axis =-1)
      logits = tf.nn.sigmoid(logits_1)
      xe = tf.reduce_mean(tf.square(logits - labels))
      return xe, logits, logits_1

  return snt.Module(_pre_probe)(prediction)
def pred_loss_1(prediction, labels):

    prediction = tf.squeeze(prediction, -1)
    labels = tf.squeeze(labels, 0)

    xe = tf.reduce_mean(tf.abs(prediction - labels))

    return xe

def mse_metric(target, pre):
    rec_mse_loss, mse = tf.unstack(pre)

    return mse

def pre_mse(target, pre):

    best_pre_loss = pre

    return best_pre_loss
