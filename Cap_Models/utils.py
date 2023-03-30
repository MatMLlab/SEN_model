
from pymatgen.core import Structure
from copy import deepcopy
from typing import Dict, List, Union
import numpy as np
from models.layers.atom_env import GraphBatchDistanceConvert, GraphBatchGenerator, StructureGraphFixedRadius
from models.layers.preprocessing import DummyScaler, Scaler
from submodels import get_graphs_within_cutoff

def data_pro(structure_data, crystal_graph):

    for idx_task, task in enumerate(structure_data.tasks):
        task.load()
        for i, fold in enumerate(task.folds):
            train_struc_data, train_bandgap_data = task.get_train_and_val_data(fold)
            test_struc_data, test_bandgap_data = task.get_test_data(fold, include_target=True)


    material_ids = []
    train_tar = []
    for i, j in train_bandgap_data.items():
        train_tar.append(j)
    train_tar = np.array(train_tar)

    train_targets = []
    train_graphs = []
    n = 0
    for i, j in train_struc_data.items():
        index1, index2, _, bonds = get_graphs_within_cutoff(j, cutoff = 5)
        if np.size(np.unique(index1)) != 0:
            graph = crystal_graph.convert(j)
            train_graphs.append(graph)
            train_targets.append(train_tar[n])
            material_ids.append(n)
            n = n + 1
        else:
            composition = j.formula
            n = n + 1
            print(composition)

    train_num = len(train_graphs)
    test_tar = []
    for i, j in test_bandgap_data.items():
        test_tar.append(j)
    test_tar = np.array(test_tar)


    n = 0
    for i, j in test_struc_data.items():
        index1, index2, _, bonds = get_graphs_within_cutoff(j, cutoff = 5)
        if np.size(np.unique(index1)) != 0:
            graph = crystal_graph.convert(j)
            train_graphs.append(graph)
            train_targets.append(test_tar[n])
            material_ids.append(n + train_num)
            n = n + 1
        else:
            composition = j.formula
            n = n + 1
            print(composition)

    max_bp = np.max(np.array(train_targets))
    #min_bp = np.min(np.array(train_targets))
    train_targets = np.array(train_targets)/max_bp
    #test_targets = np.array(test_targets)

    final_graphs = {i: j for i, j in zip(material_ids, train_graphs)}
    final_targets = {i: j for i, j in zip(material_ids, train_targets)}

    from sklearn.model_selection import train_test_split

    train_ids, test_ids = train_test_split(material_ids, test_size=0.2, shuffle = True, random_state = 66)
    train_ids = np.array(train_ids)
    test_ids = np.array(test_ids)

    ## Get the train, val and test graph-target pairs
    def get_graphs_targets(ids):

        ids = [i for i in ids if i in final_graphs]
        return [final_graphs[i] for i in ids], [final_targets[i] for i in ids]

    train_graphs, train_targets = get_graphs_targets(train_ids)
    test_graphs, test_targets = get_graphs_targets(test_ids)

    return train_graphs, train_targets, test_graphs, test_targets



class CryMat_Gen(object):
    """
    Make the generator for keras.fit_generator
    """
    def __init__(self,
                 train_graphs: List[Dict] = None,
                 train_targets: List[float] = None, batch_size = 16,
                 sample_weights: List[float] = None, mode = 'train',
                 val_graphs: List[Dict] = None,
                 val_targets: List[float] = None,
                 target_scaler: Scaler = DummyScaler(),
                 graph_converter = StructureGraphFixedRadius,
                 scrub_failed_structures: bool = False,
                 ):
        self.train_graphs = train_graphs
        self.train_targets = train_targets
        self.batch_size = batch_size
        self.sample_weights = sample_weights
        self.mode = mode
        self.val_graphs = val_graphs
        self.val_targets = val_targets
        self.target_scaler = target_scaler
        self.graph_converter = graph_converter
        self.scrub_failed_structures = scrub_failed_structures


    def load_train_data(self):
        train_nb_atoms = [len(i["atom"]) for i in self.train_graphs]

        train_targets = [self.target_scaler.transform(i, j) for i, j in zip(self.train_targets, train_nb_atoms)]

        train_inputs = self.graph_converter.get_flat_data(self.train_graphs, train_targets)

        train_generator = self._create_generator(*train_inputs,
                                                 sample_weights=self.sample_weights,
                                                 batch_size=self.batch_size)
        steps_per_train = int(np.ceil(len(self.train_graphs) / self.batch_size))


        return train_generator, steps_per_train

    def load_val_data(self):
        val_nb_atoms = [len(i["atom"]) for i in self.val_graphs]

        val_targets = [self.target_scaler.transform(i, j) for i, j in zip(self.val_targets, val_nb_atoms)]

        val_inputs = self.graph_converter.get_flat_data(self.val_graphs, val_targets)

        val_generator = self._create_generator(*val_inputs,
                                                 sample_weights=self.sample_weights,
                                                 batch_size=self.batch_size)
        steps_per_val = int(np.ceil(len(self.val_graphs) / self.batch_size))

        return val_generator, steps_per_val

    def _create_generator(self, *args, **kwargs):

        kwargs.update({"distance_converter": self.graph_converter.bond_converter})
        return GraphBatchDistanceConvert(*args, **kwargs)









