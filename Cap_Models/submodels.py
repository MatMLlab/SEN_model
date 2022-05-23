"""
Functional modules
Operation utilities on lists and arrays
"""
from typing import Callable, Any
from monty.json import MSONable
#import tensorflow.compat.v1.keras.backend as kb
from tensorflow.compat.v1.keras.activations import deserialize, serialize
from tensorflow.compat.v1.keras.activations import get as keras_get
from tensorflow.compat.v1.keras.layers import Dense, Dropout

from tensorflow.compat.v1.keras.layers import Layer
import numpy as np
from models.atom_env_block import atom_che_env
from pymatgen.core import Structure, Molecule
from pymatgen.optimization.neighbors import find_points_in_spheres

from models.layers.config import DataType
from typing import Union, List
from collections import Iterable
import tensorflow.compat.v1 as tf




def expand_dims(tensor, axis, n_dims=1):

  for _ in range(n_dims):
    tensor = tf.expand_dims(tensor, axis)
  return tensor

def make_brodcastable(tensor, against_tensor):
  n_dim_diff = against_tensor.shape.ndims - tensor.shape.ndims
  assert n_dim_diff >= 0
  return expand_dims(tensor, axis=-1, n_dims=n_dim_diff)


def dense(input, units=None):
    out = input
    for i in units:
        out = Dense(i, activation = tf.nn.selu)(out)
    return out





class Converter(MSONable):


    def convert(self, d: Any) -> Any:

        raise NotImplementedError

def get_graphs_within_cutoff( structure, cutoff: float = 5.0, numerical_tol: float = 1e-8):

    if isinstance(structure, Structure):
        lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
        pbc = np.array([1, 1, 1], dtype=int)
    elif isinstance(structure, Molecule):
        lattice_matrix = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]], dtype=float)
        pbc = np.array([0, 0, 0], dtype=int)
    else:
        raise ValueError("structure type not supported")
    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords, cart_coords, r=r, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
    )
    center_indices = center_indices.astype(DataType.np_int)
    neighbor_indices = neighbor_indices.astype(DataType.np_int)
    images = images.astype(DataType.np_int)
    distances = distances.astype(DataType.np_float)
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    return center_indices[exclude_self], neighbor_indices[exclude_self], images[exclude_self], distances[exclude_self]


def expand_1st(x: np.ndarray) -> np.ndarray:

    return np.expand_dims(x, axis=0)

def to_list(x: Union[Iterable, np.ndarray]) -> List:

    if isinstance(x, Iterable):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()  # noqa
    return [x]

def transformer(atom_inp, bond_inp, state_inp, number,
                    com_w, ele_idx, atom_fea, fea, nbr,
                    atom_index, nei_index, atom_sou, bond_sou, dropout = False):
    if number == 0:
        atom_vec_ = dense(atom_inp, units=[128, 192])
        bond_vec_ = dense(bond_inp, units=[128, 192])
        state_vec_ = dense(state_inp, units=[128, 192])
    else:
        atom_vec_ = atom_inp
        bond_vec_ = bond_inp
        state_vec_ = state_inp
    rep_out = atom_che_env([128, 128, 80], [128, 128, 64], [128, 128, 64],
                            activation=tf.nn.selu)([com_w, ele_idx, atom_fea, fea, nbr, atom_vec_, bond_vec_,
                                                       state_vec_, atom_index, nei_index, atom_sou, bond_sou])

    atom_che, bond_che, state_che, comp_che = rep_out[0], rep_out[1], rep_out[2], rep_out[3]

    if dropout:
        atom_che = Dropout(dropout)(atom_che)
        bond_che = Dropout(dropout)(bond_che)
        state_che = Dropout(dropout)(state_che)
    return atom_che, bond_che, state_che, comp_che



class GaussianExpansion(Layer):


    def __init__(self, centers, width, **kwargs):

        self.centers = np.array(centers).ravel()
        self.width = width
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        build the layer
        Args:
            input_shape (tuple): tuple of int for the input shape
        """
        self.built = True

    def call(self, inputs, masks=None):

        return tf.math.exp(-((inputs[:, :, None] - self.centers[None, None, :]) ** 2) / self.width ** 2)

    def compute_output_shape(self, input_shape):

        return input_shape[0], input_shape[1], len(self.centers)

    def get_config(self):

        base_config = super().get_config()
        config = {"centers": self.centers.tolist(), "width": self.width}
        return dict(list(base_config.items()) + list(config.items()))




