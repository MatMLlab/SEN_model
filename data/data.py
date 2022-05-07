"""
Prepare data from Material Project, such as bandgap, cif, formation energy ...
"""

import numpy as np
import json
from pymatgen import MPRester
import itertools
import random


API_KEY = '...' # personal license

elements0 = ['H', 'Li', 'Na', 'K', 'Rb','Cs',
             'Ba', 'Sr', 'Ca', 'Mg', 'Be',
             'La', 'Sc', 'Y', 'Ti', 'Zr', 'Hf','Ta', 'W', 'Mo',
             'Re','Tc','Mn','Cr', 'V','Nb',
             'Fe','Co', 'Ni','Pd','Pt','Os', 'Ir', 'Ru', 'Rh',
             'Cu', 'Zn','Ag','Au', 'Hg','Cd',
             'B', 'Al', 'Ga', 'In', 'Tl',
             'C', 'Si', 'Ge', 'Sn', 'Pb',
             'N', 'P', 'As', 'Sb', 'Bi',
             'O', 'S', 'Se', 'Te',
             'F', 'Cl', 'Br','I', 'Xe']


elements1 = ["H", "Li", "Be","C", "N", "O", "Na",
            "Mg", "Al", "Si", "P", "S", "K", "Ca", "Sc", "Ti",
            "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
            "Se", "Br", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
            "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "Cs",
            "Ba", "La", "Ta", "W", "Re", "Os"]

all_symbols = len(elements0)
complx = list(itertools.combinations(elements0, 3))

complx_picked=random.sample(complx,300)

system=list(map(lambda x: complx_picked[x][0] + '-' + complx_picked[x][1] + '-' + complx_picked[x][2],
                np.arange(len(complx_picked))))

property=["unit_cell_formula", "pretty_formula", "spacegroup", "energy", "erengy_per_atom",
          "volume", "formation_energy_per_atom", "nsites", "unit_cell_formula", "is_hubbard",
          "elements", "nelements", "e_above_hull", "hubbards", "is_compatible",
          "band_gap", "density", "icsd_id", "cif", "material_id"]

def que(x):
    a = MPRester(API_KEY)
    data = a.query(criteria=x,properties=property)
    return data

data=[]
valid_num=[]
for i in system:
    res=que(i)
    if res !=[]:
        data.extend(res)
        valid_num.append(system.index(i))
        print(len(res),system.index(i))


filename='data.json'
with open(filename,'w') as file_obj:
    json.dump(data,file_obj)



print("SUCCESSFUL")
