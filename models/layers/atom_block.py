
from typing import Dict, Sequence

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import regularizers, constraints, initializers
from tensorflow.compat.v1.keras.layers import Layer



class atom_env_transformer(Layer):


    def __init__(
        self,
        activation = None,
        use_bias = True,
        kernel_initializer = "glorot_uniform",
        bias_initializer = "zeros",
        kernel_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        bias_constraint = None,
        **kwargs):

        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super().__init__(**kwargs)

    def call(self, inputs):

        stoi_p = self.phi_u_h(inputs)
        e_p = self.phi_e(inputs)
        b_ei_p = self.rho_e_v(e_p, inputs)
        v_p = self.phi_v(b_ei_p, inputs)
        b_e_p = self.rho_e_u(e_p, inputs)
        b_v_p = self.rho_v_u(v_p, inputs)
        u_p = self.phi_u(b_e_p, b_v_p, inputs)
        return [v_p, e_p, u_p, stoi_p]

    def phi_u_h(self,inputs):

        raise NotImplementedError

    def phi_e(self, inputs):

        raise NotImplementedError

    def rho_e_v(self, e_p, inputs):

        raise NotImplementedError

    def phi_v(self, b_ei_p, inputs):

        raise NotImplementedError

    def rho_e_u(self, e_p, inputs):

        raise NotImplementedError

    def rho_v_u(self, v_p, inputs):

        raise NotImplementedError

    def phi_u(self, b_e_p, b_v_p, inputs):

        raise NotImplementedError

    def get_config(self) -> Dict:

        config = {
            "activation":serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))  # noqa
