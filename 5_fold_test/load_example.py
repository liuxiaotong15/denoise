import json
import random
import os
import sys

from pymatgen.core.structure import Structure

seed = 123

data_path = 'ordered_0.json'
with open(data_path,'r') as fp:
    d = json.load(fp)

print("ordered exp structure number: {n}".format(n=len(d)))

s_exp = [Structure.from_dict(x['structure']) for x in d.values()]
t_exp = [x['band_gap'] for x in d.values()]

for i in range(len(s_exp)):
    s_exp[i].remove_oxidation_states()
    s_exp[i].state=[0]

