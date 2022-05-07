
from inspect import getfullargspec
from typing import Dict, List, Union

from pymatgen.core import Structure, Molecule
from pymatgen.analysis import local_env
from pymatgen.analysis.local_env import (
    NearNeighbors,
    VoronoiNN,
    JmolNN,
    MinimumDistanceNN,
    OpenBabelNN,
    CovalentBondNN,
    MinimumVIRENN,
    MinimumOKeeffeNN,
    BrunnerNN_reciprocal,
    BrunnerNN_real,
    BrunnerNN_relative,
    EconNN,
    CrystalNN,
    CutOffDictNN,
    Critic2NN,
)


class MinimumDistanceNNAll(NearNeighbors):


    def __init__(self, cutoff: float = 4.0):

        self.cutoff = cutoff

    def get_nn_info(self, structure: Structure, n: int) -> List[Dict]:


        site = structure[n]
        neighs_dists = structure.get_neighbors(site, self.cutoff)

        siw = []
        for nn in neighs_dists:
            siw.append(
                {
                    "site": nn,
                    "image": self._get_image(structure, nn),
                    "weight": nn.nn_distance,
                    "site_index": self._get_original_site(structure, nn),
                }
            )
        return siw


class AllAtomPairs(NearNeighbors):


    def get_nn_info(self, molecule: Molecule, n: int) -> List[Dict]:

        site = molecule[n]
        siw = []
        for i, s in enumerate(molecule):
            if i != n:
                siw.append({"site": s, "image": None, "weight": site.distance(s), "site_index": i})
        return siw


def serialize(identifier: Union[str, NearNeighbors]):

    if isinstance(identifier, str):
        return identifier
    if isinstance(identifier, NearNeighbors):
        args = getfullargspec(identifier.__class__.__init__).args
        d = {"@module": identifier.__class__.__module__, "@class": identifier.__class__.__name__}
        for arg in args:
            if arg == "self":
                continue
            try:
                a = identifier.__getattribute__(arg)
                d[arg] = a
            except AttributeError:
                raise ValueError("Cannot find the argument")
        if hasattr(identifier, "kwargs"):
            d.update(**identifier.kwargs)
        return d
    if identifier is None:
        return None

    raise ValueError("Unknown identifier for local environment ", identifier)


def deserialize(config: Dict):

    if config is None:
        return None
    if ("@module" not in config) or ("@class" not in config):
        raise ValueError("The config dict cannot be loaded")
    modname = config["@module"]
    classname = config["@class"]
    mod = __import__(modname, globals(), locals(), [classname])
    cls_ = getattr(mod, classname)
    data = {k: v for k, v in config.items() if not k.startswith("@")}
    return cls_(**data)


NNDict = {
    i.__name__.lower(): i
    for i in [
        NearNeighbors,
        VoronoiNN,
        JmolNN,
        MinimumDistanceNN,
        OpenBabelNN,
        CovalentBondNN,
        MinimumVIRENN,
        MinimumOKeeffeNN,
        BrunnerNN_reciprocal,
        BrunnerNN_real,
        BrunnerNN_relative,
        EconNN,
        CrystalNN,
        CutOffDictNN,
        Critic2NN,
        MinimumDistanceNNAll,
        AllAtomPairs,
    ]
}


def get(identifier: Union[str, Dict, NearNeighbors]) -> NearNeighbors:

    if isinstance(identifier, str):
        if identifier.lower() in NNDict:
            return NNDict.get(identifier.lower())

        nn = getattr(local_env, identifier, None)
        if nn is not None:
            return nn

    if isinstance(identifier, dict):
        return deserialize(identifier)

    if isinstance(identifier, NearNeighbors):
        return identifier

    raise ValueError("%s not identified" % str(identifier))
