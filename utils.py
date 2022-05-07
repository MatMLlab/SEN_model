"""
This is the data process for material data and all data solution.
"""

from pymatgen.core import Structure
from copy import deepcopy
from typing import Dict, List, Union
import numpy as np
from models.layers.graph import GraphBatchDistanceConvert, GraphBatchGenerator, StructureGraph
from models.layers.preprocessing import DummyScaler, Scaler


def data_pro(bandgap_data, mode, structure_data, crystal_graph):

    useful_ids = set.union(*[set(bandgap_data[i].keys()) for i in mode])
    print('Only %d structures are used' % len(useful_ids))
    print('Calculating the graphs for all structures... this may take minutes.')
    structure_data = {i: structure_data[i] for i in useful_ids}
    structure_data = {i: crystal_graph.convert(Structure.from_str(j, fmt='cif'))
                      for i, j in structure_data.items()}

    ##  Generate graphs with fidelity information
    graphs = []
    targets = []
    material_ids = []

    for fidelity_id, fidelity in enumerate(mode):
        for mp_id in bandgap_data[fidelity]:
            if bandgap_data[fidelity][mp_id] < 15 :
                targets.append(bandgap_data[fidelity][mp_id])

                graph = deepcopy(structure_data[mp_id])
                # The fidelity information is included here by changing the state attributes
                # PBE: 0, GLLB-SC: 1, HSE: 2, SCAN: 3
                graph['state'] = [fidelity_id]
                # graph['target'] = [bandgap_data[fidelity][mp_id]]
                graphs.append(graph)
            else:
                pass
            # the new id is of the form mp-id_fidelity, e.g., mp-1234_pbe
            material_ids.append('%s_%s' % (mp_id, fidelity))

    max_bp = np.max(np.array(targets))
    targets = targets / max_bp

    #material_ids = material_ids[:200]

    final_graphs = {i: j for i, j in zip(material_ids, graphs)}
    final_targets = {i: j for i, j in zip(material_ids, targets)}

    #  Data splits
    ##  Random seed
    SEED = 42

    ##  train:val:test = 8:1:1
    fidelity_list = [i.split('_')[1] for i in material_ids]

    train_ids = material_ids[:5600]
    test_ids = material_ids[5000:]
    #train_val_ids, test_ids = train_test_split(material_ids, test_size=0.1,
    #                                           shuffle = False, random_state=None)
    #fidelity_list = [i.split('_')[1] for i in train_val_ids]
    #train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1 / 0.9,
    #                                      shuffle = False, random_state=None)

    ##  remove pbe from validation
    #val_ids = [i for i in val_ids if not i.endswith('pbe')]

    print("Train, val and test data sizes are ", len(train_ids), len(test_ids))

    ## Get the train, val and test graph-target pairs
    def get_graphs_targets(ids):
        """
        Get graphs and targets list from the ids

        Args:
            ids (List): list of ids

        Returns:
            list of graphs and list of target values
        """
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
                 graph_converter = StructureGraph,
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


        #try:
        #    if self.mode == 'train':
        #        self.load_train_data()
        #    else:
        #        self.load_val_data()
        #except KeyError:
        #    raise KeyError('Data loader failed: choose `other mode` to load' ' data preprocessed.')


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

    def _create_generator(self, *args, **kwargs) -> Union[GraphBatchDistanceConvert, GraphBatchGenerator]:
        if hasattr(self.graph_converter, "bond_converter"):
            kwargs.update({"distance_converter": self.graph_converter.bond_converter})
            return GraphBatchDistanceConvert(*args, **kwargs)
        return GraphBatchGenerator(*args, **kwargs)









