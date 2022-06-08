import json
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import os
import gc

# To fix a MEGNet 1.2.9 bug, "cp /home/ubuntu/code/o2u_qm9/o2u_venv/lib/python3.8/site-packages/megnet/data/graph.py" to replace same file
from megnet.data.crystal import CrystalGraph, CrystalGraphDisordered
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel

import sys

seed = 123
GPU_seed = 11111

################### need to modified before running #############
GPU_device = "0"
load_old_model_enable = False
cut_value = 0.3
#################################################################

special_path = 'init_randomly_EGPHS_EPHS_EHS_EH_E' ### not whole tree, only one path.

last_commit_id = 'none'
old_model_name = '{0}_{1}_{2}_{3}.hdf5'.format(last_commit_id, GPU_device, seed, cut_value)
items = ['gllb-sc', 'pbe', 'scan', 'hse']
fidelity_state_dict = {'gllb-sc': 4, 'pbe': 3, 'scan': 2, 'hse': 1}
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(GPU_seed)
commit_id = str(os.popen('git --no-pager log -1 --oneline --pretty=format:"%h"').read())
dump_model_name = '{0}_{1}_{2}_{3}'.format(commit_id, GPU_device, seed, cut_value)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_device

ep = 5000
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
db_short_full_dict = {'G': 'gllb-sc', 'H': 'hse', 'S': 'scan', 'P': 'pbe', 'E': 'E1'}


import logging
root_logger = logging.getLogger()
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename=dump_model_name+".log",
        format='%(asctime)s-%(pathname)s[line:%(lineno)d]-%(levelname)s: %(message)s',
        level=logging.INFO)

## start to load DFT data ##
structures = {}
targets = {}

test_structures = []
test_targets = []

s_exp_disordered = []
t_exp_disordered = []

from pymatgen.core.structure import Structure
from collections import Counter
for it in items:
    structures[it] = []
    targets[it] = []
    csv_name = '../data/' + it + '_cif.csv'
    df = pd.read_csv(csv_name)
    r = list(range(len(df)))
    random.shuffle(r)
    for i in r:
        tmp = Structure.from_str(df[it+'_structure'][i], fmt='cif')
        tmp.remove_oxidation_states()
        tmp.state=[fidelity_state_dict[it]]
        structures[it].append(tmp)
        targets[it].append(df[it+'_gap'][i])
## load DFT data finished ##

# data preprocess part
if load_old_model_enable:
    import pickle
    # load the past if needed
    model = MEGNetModel.from_file(old_model_name)
    for it in items:
        error_lst = []
        prediction_lst = []
        targets_lst = []
        for i in range(len(structures[it])):
            prdc = model.predict_structure(structures[it][i]).ravel()
            tgt = targets[it][i]
            prediction_lst.append(prdc)
            targets_lst.append(tgt)
            e = (prdc - tgt)
            error_lst.append(e)
            if abs(e) > cut_value:
                targets[it][i] = prdc
# data preprocess finished

### load disorder data ####
data_path = 'disordered.json'
with open(data_path,'r') as fp:
    d = json.load(fp)

s_exp_disordered = [Structure.from_dict(x['structure']) for x in d.values()]
t_exp_disordered = [x['band_gap'] for x in d.values()]
for i in range(len(s_exp_disordered)):
    s_exp_disordered[i].remove_oxidation_states()
    s_exp_disordered[i].state=[0]
### load disorder data ####

def prediction(model, structures, targets):
    MAE = 0
    test_size = len(structures)
    for i in range(test_size):
        model_output = model.predict_structure(structures[i]).ravel()
        err = abs(model_output - targets[i])
        MAE += err
    MAE /= test_size
    return MAE

def load_exp_data(test_idx):
    train_structures, train_targets, test_strcts, test_tgts = [], [], [], []
    for fold_i in range(5):
        data_path = 'ordered_{0}.json'.format(fold_i)
        with open(data_path,'r') as fp:
            d = json.load(fp)
        s_exp = [Structure.from_dict(x['structure']) for x in d.values()]
        t_exp = [x['band_gap'] for x in d.values()]
        for i in range(len(s_exp)):
            s_exp[i].remove_oxidation_states()
            s_exp[i].state=[0]

        if fold_i  == test_idx:
            test_strcts = s_exp
            test_tgts = t_exp
        else:
            train_structures.extend(s_exp)
            train_targets.extend(t_exp)
    # no need to shuffle, we have shuffled in the splitting script
    return train_structures, train_targets, test_strcts, test_tgts


def construct_dataset_from_str(db_short_str):
    s = []
    t = []
    for i in range(len(db_short_str)):
        s.extend(structures[db_short_full_dict[db_short_str[i]]])
        t.extend(targets[db_short_full_dict[db_short_str[i]]])
    c = list(zip(s, t))
    random.shuffle(c)
    s, t = zip(*c)
    return s, t

def find_sub_tree(cur_tag, history_tag):
    ###### load model #######
    father_model_name = dump_model_name + '_' + history_tag + '.hdf5'
    history_tag += '_'
    history_tag += cur_tag
    if special_path != '' and history_tag not in special_path:
        return
    else:
        pass

    cur_model_name = dump_model_name + '_' + history_tag + '.hdf5'
    cur_model = MEGNetModel.from_file(father_model_name)
    ###### get dataset ######
    s, t = construct_dataset_from_str(cur_tag)
    l = len(s)
    ###### train ############
    try:
        cur_model.train(s[:int(0.8*l)], 
                    t[:int(0.8*l)],
                    validation_structures=s[int(0.8*l):],
                    validation_targets=t[int(0.8*l):],
        # cur_model.train(test_structures,
        #             test_targets,
                    callbacks=[callback],
                    save_checkpoint=False,
                    automatic_correction=False,
                    batch_size = 256,
                    epochs=ep)
    except TypeError:
        logging.info('MAE of {tag} is: {mae}'.format(tag=history_tag, mae='nan'))
    else:
        mae = prediction(cur_model, test_structures, test_targets)
        logging.info('Ordered structures MAE of {tag} is: {mae}'.format(tag=history_tag, mae=mae))
        mae = prediction(cur_model, s_exp_disordered, t_exp_disordered)
        logging.info('Disordered structures MAE of {tag} is: {mae}'.format(tag=history_tag, mae=mae))
    cur_model.save_model(cur_model_name)
    ###### next level #######
    if len(cur_tag) > 1:
        for i in range(len(cur_tag)):
            next_tag = cur_tag[:i] + cur_tag[i+1:]
            find_sub_tree(next_tag, history_tag)
    else:
        pass


def main():
    global dump_model_name, structures, targets, test_structures, test_targets
    for i in range(5):
        logging.info('Fold-{0} start'.format(i))
        model = MEGNetModel(nfeat_edge=100, nfeat_node=16, ngvocal=1, global_embedding_dim=16, graph_converter=CrystalGraphDisordered(bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5)))
        
        # model = MEGNetModel(nfeat_edge=10, nfeat_global=2, graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))
        dump_model_name += "fold{0}".format(i)
        model.save_model(dump_model_name+'_init_randomly' + '.hdf5')
        init_model_tag = 'EGPHS'
        start_model_tag = 'EGPHS'
        
        structures['E1'], targets['E1'], test_structures, test_targets = load_exp_data(i)
        logging.info('E1 size: {0}, test size: {1}'.format(len(structures['E1']), len(test_structures)))
        
        ordered_energy = prediction(model, test_structures, test_targets)
        disordered_energy = prediction(model, s_exp_disordered, t_exp_disordered)
    
        logging.info('Prediction before trainnig, MAE of ordered: {ordered}; disordered: {disordered}.'.format(
        ordered=ordered_energy, disordered=disordered_energy))
    
        find_sub_tree(init_model_tag, 'init_randomly')


if __name__ == "__main__":
    main()
