"""
Building chemical environments for material .
"""
import sonnet as snt
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Add, Lambda, Activation, \
    Concatenate, LSTM, Multiply, Dropout

from models.set_tran_block import set_trasformer
from .cap_att_block import QKVAttention
from submodels import *

def MatCheCon(ele_inp, mat_inp, index_inp, real_bp, n_loop = 3, alpha = 0.5):

    # input data
    com_w, ele_idx, atom_fea, fea, nbr = ele_inp
    atom_vec_, bond_vec_, state_vec_ = mat_inp
    atom_index, nei_index, atom_sou, bond_sou = index_inp

    #material chemical environment construction
    def chemical_tansform(final_vec):

        total_fea = Lambda(_concat_nbrs, output_shape=_concat_nbrs_output_shape)(final_vec)
        #total_fea = Lambda(_concat_nbrs, output_shape=_concat_nbrs_output_shape)([atom_fea, bond_fea, nbr_list])
        # total_fea shape (None, N, M, 2 * atom_fea_len + bond_fea_len)
        #nbr_core = BatchNormalization(axis=-1)(total_fea)

        #nbr_vec = LSTM(512, return_sequences=True)(total_fea)
        nbr_vec = Dropout(0.2)(total_fea)
        #nbr_vec = LSTM(256, return_sequences=True)(nbr_vec)

        nbr_vec = Lambda(lambda x: tf.squeeze(x, 0))(nbr_vec)
        batch_shape = tf.shape(nbr_vec)[0]
        lstm_layers = snt.LSTM(hidden_size = 256)
        initial_state = lstm_layers.initial_state(batch_shape)
        out_seq, final_sta = lstm_layers(nbr_vec, initial_state)

        #nbr_vec = snt.BatchApply(snt.Linear(256, name='che_out'))(full_fea)
        #nbr_vec = dense(total_fea, units=[64, 256])

        return out_seq

    crystal_trans_layer = set_trasformer(loop = n_loop, n_hidden = 128)

    atom_vec_ = dense(atom_vec_, units=[64, 192])
    bond_vec_ = dense(bond_vec_, units=[64, 192])
    state_vec_ = dense(state_vec_, units=[64, 192])

    #comps = 0
    for i in range(n_loop):
        atom_vec_1 = atom_vec_
        bond_vec_1 = bond_vec_
        state_vec_1 = state_vec_
        atom_vec_1, bond_vec_1, state_vec_1, x_comp = transformer(atom_vec_1, bond_vec_1, state_vec_1, i,
                                                                  com_w, ele_idx, atom_fea, fea, nbr,
                                                                  atom_index, nei_index, atom_sou, bond_sou)

        # residual connection
        atom_vec_ = Add()([atom_vec_, alpha * atom_vec_1])
        bond_vec_ = Add()([bond_vec_, alpha * bond_vec_1])
        state_vec_ = Add()([state_vec_, alpha * state_vec_1])
        x_comp += x_comp

    # build crystal vector based on atom and bond information
    atom_vec = crystal_trans_layer([atom_vec_, atom_sou])

    bond_vec = crystal_trans_layer([bond_vec_, bond_sou])
    #comps = Concatenate(axis = -1)(comps)
    #comps = Dense(64)(comps)
    comps = Activation('softmax')(x_comp)

    x_comp = Lambda(lambda x: tf.expand_dims(x, 0))(comps)

    atom_inp = Multiply()([x_comp, atom_vec])
    #atom_inp_1 = Lambda(lambda x: tf.expand_dims(x, 0), name = 'atom_inp_1')(atom_inp)
    final_vec = Concatenate(axis = -1)([atom_inp, bond_vec, state_vec_])
    #final_vec_1 = Concatenate(axis=-1)([node_vec, edge_vec, x3_]) # 32, 1, 128

    # build the chemical environment of material based atom, element, bond features
    for _ in range(1):
       final_vec = chemical_tansform(final_vec)

    #final_vec = Lambda(lambda x:tf.squeeze(x, 0))(final_vec)

    return final_vec, real_bp

def _concat_nbrs(inps):

    # extraction of atom correlation by nbr_attention
    nbr_att_model = QKVAttention()
    atom_nbr_fea = nbr_att_model(inps, inps, inps)

    #atom_self_fea = tf.tile(tf.expand_dims(atom_fea, 2), [1, 1, 1, 1])
    full_fea = tf.concat([atom_nbr_fea, inps], -1)

    return full_fea

def _concat_nbrs_output_shape(input_shapes):

    B = input_shapes[0]
    N = input_shapes[1]
    M = input_shapes[2]

    return (B, N, 2*M)