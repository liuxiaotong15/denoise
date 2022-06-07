import json
import random
import os
import sys
seed = 123

data_path = '../data/all_data.json'
with open(data_path,'r') as fp:
    d = json.load(fp)

print("ordered exp structure number: {n}".format(n=len(d['ordered_exp'])))
print("disordered exp structure number: {n}".format(n=len(d['disordered_exp'])))

idx_lst = list(range(len(d['ordered_exp'])))
random.shuffle(idx_lst)

n_fold = 5

ret = [{} for i in range(n_fold)]

for i in range(len(idx_lst)):
    k = list(d['ordered_exp'].keys())[idx_lst[i]]
    ret[i%n_fold][k] = d['ordered_exp'][k]

for i in range(n_fold):
    filename = "ordered_{i}.json".format(i=i)
    with open(filename, 'w') as f:
        json.dump(ret[i], f)

for i in range(n_fold):
   print(len(ret[i]))

filename = "disordered.json"
with open(filename, 'w') as f:
    json.dump(d['disordered_exp'], f)

