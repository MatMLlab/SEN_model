import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as kb
from tensorflow.compat.v1.keras import activations, initializers, regularizers, constraints
from tensorflow.compat.v1.keras.layers import Layer

class set_trasformer(Layer):
    """
    arXiv:1511.06391 (2015).
    """

    def __init__(
        self,
        loop=3,
        n_hidden=512,
        activation=None,
        activation_lstm="selu",
        recurrent_activation="hard_sigmoid",
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        use_bias=True,
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.activation_lstm = activations.get(activation_lstm)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.unit_forget_bias = unit_forget_bias
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.loop = loop
        self.n_hidden = n_hidden

    def build(self, input_shape):
        feature_shape, index_shape = input_shape
        self.m_weight = self.add_weight(
            shape=(int(feature_shape[-1]), self.n_hidden),
            initializer=self.kernel_initializer,
            name="x_to_m_weight",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.m_bias = self.add_weight(
                shape=(self.n_hidden,),
                initializer=self.bias_initializer,
                name="x_to_m_bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.m_bias = None

        self.recurrent_kernel = self.add_weight(
            shape=(2 * self.n_hidden, 4 * self.n_hidden),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return kb.concatenate(
                        [
                            self.bias_initializer((self.n_hidden,), *args, **kwargs),
                            initializers.Ones()((self.n_hidden,), *args, **kwargs),
                            self.bias_initializer((self.n_hidden * 2,), *args, **kwargs),
                        ]
                    )

            else:
                bias_initializer = self.bias_initializer
            self.recurrent_bias = self.add_weight(
                shape=(self.n_hidden * 4,),
                name="recurrent_bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.recurrent_bias = None
        self.built = True

    def compute_output_shape(self, input_shape):
        feature_shape, index_shape = input_shape
        return feature_shape[0], None, 2 * self.n_hidden

    def call(self, inputs, mask=None):
        features, feature_graph_index = inputs
        feature_graph_index = tf.reshape(feature_graph_index, (-1,))
        _, _, count = tf.unique_with_counts(feature_graph_index)
        m = kb.dot(features, self.m_weight)
        if self.use_bias:
            m += self.m_bias

        self.h = tf.zeros(tf.stack([tf.shape(input=features)[0], tf.shape(input=count)[0], self.n_hidden]))
        self.c = tf.zeros(tf.stack([tf.shape(input=features)[0], tf.shape(input=count)[0], self.n_hidden]))
        q_star = tf.zeros(tf.stack([tf.shape(input=features)[0], tf.shape(input=count)[0], 2 * self.n_hidden]))
        for i in range(self.loop):
            self.h, c = self._lstm(q_star, self.c)
            e_i_t = tf.reduce_sum(input_tensor=m * repeat_with_index(self.h, feature_graph_index), axis=-1)
            maxes = tf.math.segment_max(e_i_t[0], feature_graph_index)
            e_i_t -= tf.expand_dims(tf.gather(maxes, feature_graph_index, axis=0), axis=0)
            exp = tf.exp(e_i_t)
            seg_sum = tf.transpose(
                a=tf.math.segment_sum(tf.transpose(a=exp, perm=[1, 0]), feature_graph_index), perm=[1, 0]
            )
            seg_sum = tf.expand_dims(seg_sum, axis=-1)
            interm = repeat_with_index(seg_sum, feature_graph_index)
            a_i_t = exp / interm[..., 0]
            r_t = tf.transpose(
                a=tf.math.segment_sum(
                    tf.transpose(a=tf.multiply(m, a_i_t[:, :, None]), perm=[1, 0, 2]), feature_graph_index
                ),
                perm=[1, 0, 2],
            )
            q_star = kb.concatenate([self.h, r_t], axis=-1)
        return q_star

    def _lstm(self, h, c):
        z = kb.dot(h, self.recurrent_kernel)
        if self.use_bias:
            z += self.recurrent_bias
        z0 = z[:, :, : self.n_hidden]
        z1 = z[:, :, self.n_hidden : 2 * self.n_hidden]
        z2 = z[:, :, 2 * self.n_hidden : 3 * self.n_hidden]
        z3 = z[:, :, 3 * self.n_hidden :]
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        # print(z.shape, f.shape, c.shape, z2.shape)
        c = f * c + i * self.activation_lstm(z2)
        o = self.recurrent_activation(z3)
        h = o * self.activation_lstm(c)
        return h, c

    def get_config(self):
        config = {
            "T": self.loop,
            "n_hidden": self.n_hidden,
            "activation": activations.serialize(self.activation),
            "activation_lstm": activations.serialize(self.activation_lstm),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": initializers.serialize(self.recurrent_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "use_bias": self.use_bias,
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(self.recurrent_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

def _repeat(x: tf.Tensor, n: tf.Tensor, axis: int = 1) -> tf.Tensor:
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


def repeat_with_index(x: tf.Tensor, index: tf.Tensor, axis: int = 1):

    index = tf.reshape(index, (-1,))
    _, _, n = tf.unique_with_counts(index)
    return _repeat(x, n, axis)