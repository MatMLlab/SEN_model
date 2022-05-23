
from abc import abstractmethod
from inspect import signature
from operator import itemgetter
import json

import numpy as np
from monty.json import MSONable
from tensorflow.compat.v1.keras.utils import Sequence
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Composition

from models.layers import atom_interavtion
from submodels import get_graphs_within_cutoff, expand_1st, to_list



class StructureGraphFixedRadius(MSONable):
    def __init__(self,
                 nn_strategy = None,
                 atom_converter = None,
                 bond_converter = None,
                 **kwargs,
                 ):


        if isinstance(nn_strategy, str):
            strategy = atom_interavtion.get(nn_strategy)
            parameters = signature(strategy).parameters
            param_dict = {i: j.default for i, j in parameters.items()}
            for i, j in kwargs.items():
                if i in param_dict:
                    setattr(self, i, j)
                    param_dict.update({i: j})
            self.nn_strategy = strategy(**param_dict)
        elif isinstance(nn_strategy, NearNeighbors):
            self.nn_strategy = nn_strategy
        elif nn_strategy is None:
            self.nn_strategy = None
        else:
            raise RuntimeError("Strategy not valid")

        self.atom_converter = atom_converter
        self.bond_converter = bond_converter


    def convert(self, structure, state_attributes = None):

        # element emebedding block
        fea_path = "/home/manager/data1/0-BandGap_Prediction_Case/NMI_Codes/SNE_Codes/models/layers/resources/matscholar-embedding.json"
        elem_features = LoadFeaturiser(fea_path)

        state_attributes = (
            state_attributes or getattr(structure, "state", None) or np.array([[0.0, 0.0]], dtype="float32")
        )
        atoms = self.get_atom_features(structure)
        index1, index2, _, bonds = get_graphs_within_cutoff(structure, self.nn_strategy.cutoff)
        composition = structure.formula
        comp_dict = Composition(composition).get_el_amt_dict()
        elements_dic = list(comp_dict.keys())
        atom_fea = np.vstack(
            [elem_features.get_fea(element) for element in elements_dic]
        )
        elements_num = len(list(comp_dict.keys()))
        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / np.sum(weights)

        if np.size(np.unique(index1)) < len(atoms):
            raise RuntimeError("Isolated atoms found in the structure")
        return {"comp_w": weights, "elem_num": elements_num, "atom_fea": atom_fea,
                "atom": atoms, "bond": bonds, "state": state_attributes,
                "index1": index1, "index2": index2}

    @classmethod
    def from_structure_graph(cls, structure_graph):

        return cls(
            nn_strategy=structure_graph.nn_strategy,
            atom_converter=structure_graph.atom_converter,
            bond_converter=structure_graph.bond_converter,
        )

    @staticmethod
    def get_flat_data(graphs, targets=None):

        output = []

        for feature in ["comp_w", "elem_num", "atom_fea", "atom", "bond", "state", "index1", "index2"]:
            output.append([np.array(x[feature]) for x in graphs])

        if targets is not None:
            output.append([to_list(t) for t in targets])

        return tuple(output)

    @staticmethod
    def get_atom_features(structure):

        return np.array([i.specie.Z for i in structure], dtype="int32").tolist()


    @classmethod
    def from_dict(cls, d):

        if "nn_strategy" in d:
            nn_strategy = d.pop("nn_strategy")
            nn_strategy_obj = atom_interavtion.deserialize(nn_strategy)
            d.update({"nn_strategy": nn_strategy_obj})
            return super().from_dict(d)
        return super().from_dict(d)

    def as_dict(self):

        all_dict = super().as_dict()
        if "nn_strategy" in all_dict:
            nn_strategy = all_dict.pop("nn_strategy")
            all_dict.update({"nn_strategy": atom_interavtion.serialize(nn_strategy)})
        return all_dict


class LoadFeaturiser(object):


    def __init__(self, embedding_file):

        self._embedding = {}

        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = set(embedding.keys())
        leng = len(allowed_types)
        ele_hot = np.eye(leng)
        i = 0
        for key, value in embedding.items():
            #self._embedding[key] = np.array(value, dtype=float)
            self._embedding[key] = ele_hot[i, :].reshape(-1)
            i += 1

    def get_fea(self, key):

        return self._embedding[key]


class GaussianDistance(object):

    def __init__(self, centers: np.ndarray = np.linspace(0, 5, 100), width=0.5):

        self.centers = centers
        self.width = width

    def convert(self, d):

        d = np.array(d)
        return np.exp(-((d[:, None] - self.centers[None, :]) ** 2) / self.width ** 2)


class BaseGraphBatchGenerator(Sequence):


    def __init__(
        self,
        dataset_size: int,
        targets: np.ndarray,
        sample_weights: np.ndarray = None,
        batch_size: int = 128,
        is_shuffle: bool = False,
    ):

        if targets is not None:
            self.targets = np.array(targets).reshape((dataset_size, -1))
        else:
            self.targets = None

        if sample_weights is not None:
            self.sample_weights = np.array(sample_weights)
        else:
            self.sample_weights = None

        self.batch_size = batch_size
        self.total_n = dataset_size
        self.is_shuffle = is_shuffle
        self.max_step = int(np.ceil(self.total_n / batch_size))
        self.mol_index = np.arange(self.total_n)

    def __len__(self) -> int:
        return self.max_step

    def _combine_graph_data(
        self,
        comp_list_temp,
        ele_list_temp,
        atom_list_temp,
        feature_list_temp,
        connection_list_temp,
        global_list_temp,
        index1_temp,
        index2_temp,
        targets,
    ) -> tuple:

        targets = np.array(targets).reshape(-1)
        gnode = []
        for i, j in enumerate(feature_list_temp):
            gnode += [i] * len(j)
        # get bond features from a batch of structures
        # get bond's structure id
        gbond = []
        for i, j in enumerate(connection_list_temp):
            gbond += [i] * len(j)

        # assemble atom features together
        feature_list_temp = np.concatenate(feature_list_temp, axis=0)
        feature_list_temp = self.process_atom_feature(feature_list_temp)

        # assemble bond feature together
        connection_list_temp = np.concatenate(connection_list_temp, axis=0)
        connection_list_temp = self.process_bond_feature(connection_list_temp)

        #assemble crystal_atom_idx feature together
        crystal_atom_idx = []
        for i, cry_id in enumerate(ele_list_temp):
            n_i = int(cry_id)
            crystal_atom_idx.append([i] * n_i)
        #crystal_atom_idx = np.array(crystal_atom_idx).reshape(-1)
        #comp_list_temp = np.array(comp_list_temp)

        # assemble state feature together
        global_list_temp = np.concatenate(global_list_temp, axis=0)
        global_list_temp = self.process_state_feature(global_list_temp)

        batch_self_fea_idx = []
        batch_nbr_fea_idx = []
        for i, env in enumerate(ele_list_temp):
            env_idx = list(range(int(env)))
            self_fea_idx = []
            nbr_fea_idx = []
            nbrs = int(env) - 1
            for i in range(int(env)):
                if nbrs == 0:
                    self_fea_idx += [i]
                else:
                    self_fea_idx += [i] * nbrs


                if env_idx == [0]:
                    nbr_fea_idx += [0]
                else:
                    nbr_fea_idx += env_idx[:i] + env_idx[i + 1:]
            batch_self_fea_idx.append(self_fea_idx)
            batch_nbr_fea_idx.append(nbr_fea_idx)

        self_fea_list = []
        cry_base_idx = 0
        for i, self_idx in enumerate(batch_self_fea_idx):
            n_i = len(set(self_idx))
            a = [i + cry_base_idx for i in self_idx]
            self_fea_list.append(a)
            cry_base_idx += n_i

        nbr_fea_list = []
        nbr_base_idx = 0
        for i, nbr_idx in enumerate(batch_nbr_fea_idx):
            n_i = len(set(nbr_idx))
            b = [i + nbr_base_idx for i in nbr_idx]
            nbr_fea_list.append(b)
            nbr_base_idx += n_i

        # assemble bond indices
        index1 = []
        index2 = []
        comp_list = []
        crystal_list = []
        fea_list = []
        nbr_list = []
        offset_ind = 0
        for ind1, ind2, clt, cai, nfl, cl in zip(index1_temp, index2_temp, comp_list_temp,
                                                 crystal_atom_idx, self_fea_list, nbr_fea_list):
            index1 += [i + offset_ind for i in ind1]
            index2 += [i + offset_ind for i in ind2]
            comp_list += [i for i in clt]
            crystal_list += [i for i in cai]
            fea_list += [i for i in nfl]
            nbr_list += [i for i in cl]
            offset_ind += max(ind1) + 1
        # Compile the inputs in needed order
        atom_list_temp = np.concatenate(atom_list_temp, axis=0)
        inputs = (expand_1st(np.array(comp_list).reshape(-1)),
                  expand_1st(np.array(crystal_list).reshape(-1)),
                  expand_1st(np.array(atom_list_temp)),
                  expand_1st(np.array(fea_list)),
                  expand_1st(np.array(nbr_list)),

                  expand_1st(feature_list_temp),
                  expand_1st(connection_list_temp),
                  expand_1st(global_list_temp),
                  expand_1st(np.array(index1, dtype=np.int32)),
                  expand_1st(np.array(index2, dtype=np.int32)),
                  expand_1st(np.array(gnode, dtype=np.int32)),
                  expand_1st(np.array(gbond, dtype=np.int32)),
                  expand_1st(np.array(targets)))
        return inputs


    def process_atom_feature(self, x):

        return x

    def process_bond_feature(self, x):

        return x

    def process_state_feature(self, x):

        return x

    def __getitem__(self, index):
        # Get the indices for this batch
        #print("index:", index)
        batch_index = self.mol_index[index * self.batch_size : (index + 1) * self.batch_size]

        # Get the inputs for each batch
        inputs = self._generate_inputs(batch_index)
        #print("shape of inputs:", np.array(inputs).shape)
        # Make the graph data
        inputs = self._combine_graph_data(*inputs)
        inputs = list(inputs)

        # Return the batch
        if self.targets is None:
            return inputs
        # get targets
        target_temp = itemgetter_list(self.targets, batch_index)
        target_temp = np.atleast_2d(target_temp)
        if self.sample_weights is None:
            target_label = expand_1st(target_temp)
            #print("target shape: ", np.array(target_label).shape)

            return inputs, target_label
        sample_weights_temp = itemgetter_list(self.sample_weights, batch_index)
        # sample_weights_temp = np.atleast_2d(sample_weights_temp)
        return inputs, expand_1st(target_temp), expand_1st(sample_weights_temp)

    @abstractmethod
    def _generate_inputs(self, batch_index):

        pass


class GraphBatchGenerator(BaseGraphBatchGenerator):


    def __init__(
        self,
        comp_weights,
        ele_nums,
        atom_fea,
        atom_features,
        bond_features,
        state_features,
        index1_list,
        index2_list,
        targets = None,
        sample_weights = None,
        batch_size = 128,
        is_shuffle = False,
    ):

        super().__init__(len(atom_features),
                         targets,
                         sample_weights = sample_weights,
                         batch_size = batch_size,
                         is_shuffle = is_shuffle,
        )
        self.comp_weights = comp_weights
        self.ele_nums = ele_nums
        self.atom_fea = atom_fea
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.state_features = state_features
        self.index1_list = index1_list
        self.index2_list = index2_list
        self.targets = targets

    def _generate_inputs(self, batch_index):

        comp_list_temp = itemgetter_list(self.comp_weights, batch_index)
        ele_list_temp = itemgetter_list(self.ele_nums, batch_index)
        atom_list_temp = itemgetter_list(self.atom_fea, batch_index)

        feature_list_temp = itemgetter_list(self.atom_features, batch_index)
        connection_list_temp = itemgetter_list(self.bond_features, batch_index)
        global_list_temp = itemgetter_list(self.state_features, batch_index)
        index1_temp = itemgetter_list(self.index1_list, batch_index)
        index2_temp = itemgetter_list(self.index2_list, batch_index)
        targets = itemgetter_list(self.targets, batch_index)

        return comp_list_temp, ele_list_temp, atom_list_temp, \
               feature_list_temp, connection_list_temp, global_list_temp, index1_temp, index2_temp, targets


class GraphBatchDistanceConvert(GraphBatchGenerator):


    def __init__(
        self,
        comp_weights,
        ele_nums,
        atom_fea,
        atom_features,
        bond_features,
        state_features,
        index1_list,
        index2_list,
        targets = None,
        sample_weights = None,
        batch_size = 128,
        is_shuffle = False,
        distance_converter = None,
    ):

        super().__init__(
            comp_weights = comp_weights,
            ele_nums = ele_nums,
            atom_fea = atom_fea,
            atom_features = atom_features,
            bond_features = bond_features,
            state_features = state_features,
            index1_list = index1_list,
            index2_list = index2_list,
            targets = targets,
            sample_weights = sample_weights,
            batch_size = batch_size,
            is_shuffle = is_shuffle,
        )
        if distance_converter is None:
            raise ValueError("Distance converter cannot be None")
        self.distance_converter = distance_converter

    def process_bond_feature(self, x):

        return self.distance_converter.convert(x)


def itemgetter_list(data_list, indices):

    it = itemgetter(*indices)
    if np.size(indices) == 1:
        return (it(data_list),)
    return it(data_list)
