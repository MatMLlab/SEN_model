
from pymatgen.core import Structure
from copy import deepcopy
from typing import Dict, List, Union
import numpy as np
from models.layers.atom_env import GraphBatchDistanceConvert, GraphBatchGenerator, StructureGraph
from models.layers.preprocessing import DummyScaler, Scaler


def data_pro(bandgap_data, mode, structure_data, crystal_graph):

    mat_ids = set.union(*[set(bandgap_data[i].keys()) for i in ['hse']])
    struc_data = {i: structure_data[i] for i in mat_ids}
    struc_data = {i: crystal_graph.convert(Structure.from_str(j, fmt='cif'))
                      for i, j in struc_data.items()}

    graphs = []
    targets = []
    material_ids = []
    mode = ['hse']
    for mat_id, cal_mode in enumerate(mode):
        for mp_id in bandgap_data[cal_mode]:
            if bandgap_data[cal_mode][mp_id] < 15 :
                targets.append(bandgap_data[cal_mode][mp_id])

                graph = deepcopy(struc_data[mp_id])
                graph['state'] = [mat_id]
                graphs.append(graph)
            else:
                pass
            material_ids.append('%s_%s' % (mp_id, cal_mode))

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

    train_ids = material_ids[:5000]
    test_ids = material_ids[5000:]
    #train_val_ids, test_ids = train_test_split(material_ids, test_size=0.1,
    #                                           shuffle = False, random_state=None)
    #fidelity_list = [i.split('_')[1] for i in train_val_ids]
    #train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1 / 0.9,
    #                                      shuffle = False, random_state=None)

    ##  remove pbe from validation
    #val_ids = [i for i in val_ids if not i.endswith('pbe')]

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









