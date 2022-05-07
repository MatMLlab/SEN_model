

import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nest

from models.layers.stoi_basic import *


class Stoi_Rep(snt.AbstractModule):
  """Performs k independent linear transformations of k vectors."""

  def __init__(
        self,
        n_target,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=256,
        elem_msg=256,
        cry_heads=3,
        cry_gate=256,
        cry_msg=256,
        out_hidden=[1024, 512, 256, 128, 64]):
    """Builds the roost module.
    """
    super(Stoi_Rep, self).__init__()
    self.n_target = n_target
    self.elem_emb_len = elem_emb_len
    self.elem_fea_len = elem_fea_len
    self.n_graph = n_graph
    self.elem_heads = elem_heads
    self.elem_gate = elem_gate
    self.elem_msg = elem_msg
    self.cry_heads = cry_heads
    self.cry_gate = cry_gate
    self.cry_msg = cry_msg
    self.out_hidden = out_hidden

    desc_dict = {
        "elem_emb_len": self.elem_emb_len,
        "elem_fea_len": self.elem_fea_len,
        "n_graph": self.n_graph,
        "elem_heads": self.elem_heads,
        "elem_gate": self.elem_gate,
        "elem_msg": self.elem_msg,
        "cry_heads": self.cry_heads,
        "cry_gate": self.cry_gate,
        "cry_msg": self.cry_msg,
        }

    #self.output_nn = ResidualNetwork(elem_fea_len, output_dim, out
    self.material_nn = DescriptorNetwork(**desc_dict)

  def _build(self, com_w, atom_fea, index1, index2, ele_idx):
    """
        Forward pass through the material_nn and output_nn
        """

    atom_fea = tf.squeeze(atom_fea, 0)
    ele_fea, _ = embedding_lookup(
        input_ids = atom_fea,
        vocab_size = 103,
        embedding_size = 128,
        initializer_range = 0.05,
        word_embedding_name = "atom_fea_embeddings",
        use_one_hot_embeddings = True)

    crys_fea = self.material_nn(
        com_w, ele_fea, index1, index2, ele_idx
    )

    # apply neural network to map from learned features to target
    return crys_fea

  def _init(self, key):
    if self.initializers:
      return self.initializers.get(key, None)

class DescriptorNetwork(snt.AbstractModule):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
    """

    def __init__(
        self, elem_emb_len, elem_fea_len=64, n_graph=3,
        elem_heads=3, elem_gate=256, elem_msg=256, cry_heads=3, cry_gate=256, cry_msg=256,
    ):
        """
        """
        super(DescriptorNetwork, self).__init__()
        self.elem_emb_len = elem_emb_len
        self.elem_fea_len = elem_fea_len
        self.n_graph = n_graph
        self.elem_heads = elem_heads
        self.elem_gate = elem_gate
        self.elem_msg = elem_msg
        self.cry_msg = cry_msg
        self.cry_heads = cry_heads
        self.cry_gate = cry_gate

        self.graphs = MessageLayer(elem_fea_len = self.elem_fea_len,
                                   elem_heads = self.elem_heads,
                                   elem_gate = self.elem_gate,
                                   elem_msg = self.elem_msg,
                                   )

        self.cry_pool = WeightedAttentionPooling(
            gate_nn = SimpleNetwork(elem_fea_len, 1, cry_gate),
            message_nn = SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg))

    def _build(self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx):


        # embed the original features into a trainable embedding space
        elem_fea = snt.Linear(self.elem_fea_len - 1)(elem_fea)

        # add weights as a node feature
        elem_fea = tf.concat([elem_fea, tf.expand_dims(tf.squeeze(elem_weights, 0), -1)], axis = -1) # (1, N, 64)

        # apply the message passing functions
        elem_weights = tf.transpose(elem_weights, [1, 0]) # (N, 1)
        #elem_fea = tf.squeeze(elem_fea, 0) # (N, 64)
        self_fea_idx = tf.squeeze(self_fea_idx, 0) # (L)
        nbr_fea_idx = tf.squeeze(nbr_fea_idx, 0) # (L)
        cry_elem_idx = tf.squeeze(cry_elem_idx, 0) # (L)
        for i in range(self.n_graph):
            elem_fea = self.graphs(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for i in range(self.cry_heads):
            out = self.cry_pool(elem_fea, index = cry_elem_idx, weights = elem_weights)
            head_fea.append(out)
        out_1 = tf.reduce_mean(tf.stack(head_fea), axis = 0)

        return out_1

    def _init(self, key):
        if self.initializers:
            return self.initializers.get(key, None)


class MessageLayer(snt.AbstractModule):
    """
    Massage Layers are used to propagate information between nodes in
    the stoichiometry graph.
    """

    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg):
        super(MessageLayer, self).__init__()
        self.elem_fea_len = elem_fea_len
        self.elem_heads = elem_heads
        self.elem_gate = elem_gate
        self.elem_msg = elem_msg

        # Pooling and Output
        self.pooling = WeightedAttentionPooling(
            gate_nn = SimpleNetwork(2 * self.elem_fea_len, 1, self.elem_gate),
            message_nn = SimpleNetwork(2 * self.elem_fea_len, self.elem_fea_len, self.elem_msg))

    def _build(self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx):

        # construct the total features for passing
        elem_nbr_weights = tf.gather(elem_weights, nbr_fea_idx, axis = 0)
        #elem_nbr_weights = elem_weights[tf.to_int64(nbr_fea_idx), :]
        elem_nbr_fea = tf.gather(elem_in_fea, nbr_fea_idx, axis = 0)
        #elem_nbr_fea = elem_in_fea[tf.to_int64(nbr_fea_idx), :]
        elem_self_fea = tf.gather(elem_in_fea, self_fea_idx, axis = 0)
        #elem_self_fea = elem_in_fea[tf.to_int64(self_fea_idx), :]
        fea = tf.concat([elem_self_fea, elem_nbr_fea], axis = -1)

        # sum selectivity over the neighbours to get elems
        #head_fea = []
        #for i in range(self.elem_heads):
        #    out = self.pooling(fea, index= self_fea_idx, weights = elem_nbr_weights)
        #    head_fea.append(out)

        # average the attention heads
        #a = tf.stack(head_fea)
        #fea = tf.reduce_mean(tf.stack(head_fea), axis = 0)
        fea = self.pooling(fea, index = self_fea_idx, weights = elem_nbr_weights)

        return fea + elem_in_fea

    def _init(self, key):
        if self.initializers:
            return self.initializers.get(key, None)



