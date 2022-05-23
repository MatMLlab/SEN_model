"""
atom chemical environment construction
"""
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as kb

from models.layers.atom_block import atom_env_transformer
from models.stoi_block import Stoi_Rep

class atom_che_env(atom_env_transformer):


    def __init__(
        self,
        units_v,
        units_e,
        units_u,
        pool_method="mean",
        activation = tf.nn.selu,
        use_bias=True,
        kernel_initializer = "glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):

        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.units_v = units_v
        self.units_e = units_e
        self.units_u = units_u
        self.pool_method = pool_method

        model_params = {
            "n_target": 1,
            "elem_emb_len": 200, #
            "elem_fea_len": 256,
            "n_graph": 3,
            "elem_heads": 3,
            "elem_gate": [256],
            "elem_msg": [256],
            "cry_heads": 3,
            "cry_gate": [256],
            "cry_msg": [256],
            "out_hidden": [1024, 512, 256, 128, 64],
        }

        self.Stoi_Rep = Stoi_Rep(**model_params)

        if pool_method == "mean":
            self.reduce_method = tf.reduce_mean
            self.seg_method = tf.math.segment_mean
        elif pool_method == "sum":
            self.reduce_method = tf.reduce_sum
            self.seg_method = tf.math.segment_sum
        else:
            raise ValueError("Pool method: " + pool_method + " not understood!")

    def build(self, input_shapes):

        vdim = int(input_shapes[5][2])
        edim = int(input_shapes[6][2])
        udim = int(input_shapes[7][2])


        with kb.name_scope(self.name):


            """
            with kb.name_scope("phi_e"):
                e_shapes = [2 * vdim + edim + udim] + self.units_e
                e_shapes = list(zip(e_shapes[:-1], e_shapes[1:]))
                self.phi_e_weights = [
                    self.add_weight(
                        shape = i,
                        name = "weight_e_%d" % j,
                        regularizer = self.kernel_regularizer,
                        constraint = self.kernel_constraint,
                    )
                    for j, i in enumerate(e_shapes)
                ]
                if self.use_bias:
                    self.phi_e_biases = [
                        self.add_weight(
                            shape=(i[-1],),
                            initializer=self.bias_initializer,
                            name="bias_e_%d" % j,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                        )
                        for j, i in enumerate(e_shapes)
                    ]
                else:
                    self.phi_e_biases = None

            with kb.name_scope("phi_v"):
                v_shapes = [edim + vdim + udim] + self.units_v
                v_shapes = list(zip(v_shapes[:-1], v_shapes[1:]))

                self.phi_v_weights = [
                    self.add_weight(
                        shape=i,
                        name="weight_v_%d" % j,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint)
                    for j, i in enumerate(v_shapes)]

                if self.use_bias:
                    self.phi_v_biases = [
                        self.add_weight(
                            shape=(i[-1],),
                            name="bias_v_%d" % j,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                        )
                        for j, i in enumerate(v_shapes)
                    ]
                else:
                    self.phi_v_biases = None

            with kb.name_scope("phi_u"):
                u_shapes = [edim + self.units_v[-1] + udim] + self.units_u
                u_shapes = list(zip(u_shapes[:-1], u_shapes[1:]))
                self.phi_u_weights = [
                    self.add_weight(
                        shape=i,
                        name="weight_u_%d" % j,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                    )
                    for j, i in enumerate(u_shapes)
                ]
                if self.use_bias:
                    self.phi_u_biases = [
                        self.add_weight(
                            shape=(i[-1],),
                            name="bias_u_%d" % j,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                        )
                        for j, i in enumerate(u_shapes)
                    ]
                else:
                    self.phi_u_biases = None
            """

        self.built = True

    def compute_output_shape(self, input_shape):

        node_feature_shape = input_shape[5]
        edge_feature_shape = input_shape[6]
        state_feature_shape = input_shape[7]
        output_shape = [
            (node_feature_shape[0], node_feature_shape[1], self.units_v[-1]),
            (edge_feature_shape[0], edge_feature_shape[1], self.units_e[-1]),
            (state_feature_shape[0], state_feature_shape[1], self.units_u[-1]),
        ]
        return output_shape

    def phi_u_h(self, inputs):

        com_w, ele_idx, atom_fea, x_fea, x_nbr, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs


        crys_fea = self.Stoi_Rep(
            com_w, atom_fea, x_fea, x_nbr, ele_idx
        )

        return crys_fea

    def phi_e(self, inputs):

        com_w, ele_idx, atom_fea, x_fea, x_nbr, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        index1 = tf.reshape(index1, (-1,))
        index2 = tf.reshape(index2, (-1,))
        fs = tf.gather(nodes, index1, axis=1)
        fr = tf.gather(nodes, index2, axis=1)
        concate_node = tf.concat([fs, fr], -1)
        u_expand = atom_index(u, gbond, axis=1)
        concated = tf.concat([concate_node, edges, u_expand], -1) # (1, ?, 16 * 16)

        concated_1 = tf.squeeze(concated, 0)
        concated_1 = tf.reshape(concated_1, [-1, 12, 16, 4])
        filter1 = tf.Variable(tf.random_normal([3, 3, 1, 24]))
        conv_concated1 = tf.nn.conv2d(concated_1, filter=filter1, strides=1, padding='SAME')

        h = tf.reshape(conv_concated1, [-1, 192, 24], name='bond_reshape')
        h, a = tf.split(h, [12, 12], -1)
        a = tf.nn.softmax(a, 1)
        h = tf.reduce_sum(h * a, -1)

        #out = tf.layers.dense(h, 64, activation=tf.nn.selu)
        out = tf.nn.selu(h)
        out = tf.expand_dims(out, 0)

        return out

    def rho_e_v(self, e_p, inputs):

        com_w, ele_idx, atom_fea, x_fea, x_nbr, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        index1 = tf.reshape(index1, (-1,))
        out = tf.expand_dims(self.seg_method(tf.squeeze(e_p), index1), axis=0)
        return out

    def phi_v(self, b_ei_p, inputs):

        com_w, ele_idx, atom_fea, x_fea, x_nbr, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        u_expand = atom_index(u, gnode, axis=1)
        concated = tf.concat([b_ei_p, nodes, u_expand], -1) # (1, ?, 16 * 13)

        concated_1 = tf.squeeze(concated, 0)
        # concated_1 = tf.reshape(concated_1, [-1, 192])
        concated_2 = tf.reshape(concated_1, [-1, 12, 16, 3])
        filter1 = tf.Variable(tf.random_normal([3, 3, 1, 24]))
        conv_concated1 = tf.nn.conv2d(concated_2, filter=filter1, strides=1, padding='SAME')

        h = tf.reshape(conv_concated1, [-1, 192, 24], name='atom_reshape')
        h, a = tf.split(h, [12, 12], -1)
        a = tf.nn.softmax(a, 1)
        h = tf.reduce_sum(h * a, -1)

        out = tf.nn.selu(h)
        #out = tf.layers.dense(h, 64, activation=tf.nn.selu)
        out = tf.expand_dims(out, 0)

        return out

    def rho_e_u(self, e_p, inputs):

        com_w, ele_idx, atom_fea, x_fea, x_nbr, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        gbond = tf.reshape(gbond, (-1,))
        out = tf.expand_dims(self.seg_method(tf.squeeze(e_p), gbond), axis=0)
        return out

    def rho_v_u(self, v_p, inputs):

        com_w, ele_idx, atom_fea, x_fea, x_nbr, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        gnode = tf.reshape(gnode, (-1,))
        out =tf.expand_dims(self.seg_method(tf.squeeze(v_p, axis=0), gnode), axis=0)
        return out

    def phi_u(self, b_e_p, b_v_p, inputs):

        concated = tf.concat([b_e_p, b_v_p, inputs[7]], -1) # (1, ?, 16 * 18)

        concated_1 = tf.squeeze(concated, 0)
        # concated_1 = tf.reshape(concated_1, [-1, 192])
        concated_2 = tf.reshape(concated_1, [-1, 12, 16, 3])
        filter1 = tf.Variable(tf.random_normal([3, 3, 1, 24]))

        conv_concated1 = tf.nn.conv2d(concated_2, filter=filter1, strides=1, padding='SAME')

        h = tf.reshape(conv_concated1, [-1, 192, 24])
        h, a = tf.split(h, [12, 12], -1)
        a = tf.nn.softmax(a, -1)
        h = tf.reduce_sum(h * a, -1)

        out = tf.nn.selu(h)
        out = tf.expand_dims(out, 0)
        #out = tf.layers.dense(h, 64, activation=tf.nn.selu)
        #out = tf.expand_dims(out, 0)
        return out

    def _mlp(self, input_, weights, biases):
        if biases is None:
            biases = [0] * len(weights)
        act = input_
        for w, b in zip(weights, biases):
            output = kb.dot(act, w) + b
            act = self.activation(output)
        return output

    def get_config(self):

        config = {
            "units_e": self.units_e,
            "units_v": self.units_v,
            "units_u": self.units_u,
            "pool_method": self.pool_method,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _repeat(x, n, axis = 1):
    # get maximum repeat length in x
    assert len(n.shape) == 1
    maxlen = tf.reduce_max(input_tensor=n)
    x_shape = tf.shape(input=x)
    x_dim = len(x.shape)
    # create a range with the length of x
    shape = [1] * (x_dim + 1)
    shape[axis + 1] = maxlen
    # tile it to the maximum repeat length, it should be of shape
    # [xlen, maxlen] now
    x_tiled = tf.tile(tf.expand_dims(x, axis + 1), tf.stack(shape))

    new_shape = tf.unstack(x_shape)
    new_shape[axis] = -1
    new_shape[-1] = x.shape[-1]
    x_tiled = tf.reshape(x_tiled, new_shape)
    # create a sequence mask using x
    # this will create a boolean matrix of shape [xlen, maxlen]
    # where result[i,j] is true if j < x[i].
    mask = tf.sequence_mask(n, maxlen)
    mask = tf.reshape(mask, (-1,))
    # mask the elements based on the sequence mask
    return tf.boolean_mask(tensor=x_tiled, mask=mask, axis=axis)

def atom_index(x, index, axis = 1):

    index = tf.reshape(index, (-1,))
    _, _, n = tf.unique_with_counts(index)
    return _repeat(x, n, axis)


