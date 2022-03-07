import json
import pandas as pd
import numpy as np
import random
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

import os
import gc

from megnet.data.crystal import CrystalGraph, CrystalGraphDisordered
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel
# from megnet.callbacks import XiaotongCB

import sys
training_mode = int(sys.argv[1])
seed = 123
GPU_seed = 11111
GPU_device = "0"
dump_prediction_cif = False
load_old_model_enable = True
predict_before_dataclean = False
training_new_model = True
contain_e1_in_every_node = False
swap_E1_test = False
tau_modify_enable = False

trained_last_time = True

if training_mode in [0, 1]:
    swap_E1_test = bool(training_mode&1)
    # special_path = 'init_randomly_EGPHS_EGPH_EGP_EG_E'  # worst1
    # special_path = 'init_randomly_EGPHS_EPHS_EPH_EH_E'  # better
    special_path = 'init_randomly_EGPHS_EPHS_EHS_EH_E'  # best
    last_commit_id = '223d078'
    if training_mode == 0:
        old_model_name = last_commit_id + '_0_123_' + special_path + '.hdf5'
        GPU_device = "0"
    elif training_mode == 1:
        old_model_name = last_commit_id + '_1_123_' + special_path + '.hdf5'
        GPU_device = "1"
    else:
        pass


tau_dict = {'pbe': 1.297, 'hse': 1.066, 'scan': 1.257, 'gllb-sc': 0.744} # P, H, S, G # min(MSE)
items = ['gllb-sc', 'pbe', 'scan', 'hse']
cut_value = 0.3

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(GPU_seed)

commit_id = str(os.popen('git --no-pager log -1 --oneline --pretty=format:"%h"').read())

dump_model_name = '{commit_id}_{training_mode}_{seed}'.format(commit_id=commit_id, 
        training_mode=training_mode,
        seed=seed)

import logging
root_logger = logging.getLogger()
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename=dump_model_name+".log",
        format='%(asctime)s-%(pathname)s[line:%(lineno)d]-%(levelname)s: %(message)s',
        level=logging.INFO)

def prediction(model, structures, targets):
    MAE = 0
    test_size = len(structures)
    for i in range(test_size):
        model_output = model.predict_structure(structures[i]).ravel()
        err = abs(model_output - targets[i])
        if dump_prediction_cif:
            name = '{ae}_{mo}_{target}.cif'.format(
                    ae=err, mo=model_output, target=targets[i])
            structures[i].to(filename=name)
        MAE += err
    MAE /= test_size
    return MAE
    # logging.info('MAE is: {mae}'.format(mae=MAE))


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_device

logging.info('onion training is running, the whole training process likes a tree, gogogo!')
logging.info('commit_id is: {cid}'.format(cid=commit_id))
logging.info('training_mode is: {tm}'.format(tm=training_mode))
logging.info('device number is: GPU_{d}'.format(d=GPU_device))
logging.info('GPU seed is: {d}'.format(d=GPU_seed))

logging.info('items is {it}'.format(it=str(items)))
logging.info('contain E1 in every node is {e}'.format(e=str(contain_e1_in_every_node)))
logging.info('trained_last_time is {e}'.format(e=str(trained_last_time)))

logging.info('tau_enable={t} and tau_dict is {td}'.format(
    t=str(tau_modify_enable), td=str(tau_dict)))

logging.info('load_old_model_enable={l}, old_model_name={omn}, cut_value={cv}'.format(
    l=load_old_model_enable, omn=old_model_name, cv=cut_value))
logging.info('swap_E1_test={b}'.format(b=str(swap_E1_test)))
logging.info('predict_before_dataclean={p}, training_new_model={t}'.format(
    p=predict_before_dataclean, t=training_new_model))


## start to load data ##
structures = {}
targets = {}

from pymatgen.core.structure import Structure
from collections import Counter
for it in items:
    structures[it] = []
    targets[it] = []
    csv_name = 'data/' + it + '_cif.csv'
    df = pd.read_csv(csv_name)
    r = list(range(len(df)))
    random.shuffle(r)
    sp_lst = []
    for i in r:
        tmp = Structure.from_str(df[it+'_structure'][i], fmt='cif')
        tmp.remove_oxidation_states()
        tmp.state=[0]
        structures[it].append(tmp)
        sp_lst.extend(list(set(structures[it][-1].species)))
        if tau_modify_enable:
            targets[it].append(df[it+'_gap'][i] * tau_dict[it])
        else:
            targets[it].append(df[it+'_gap'][i])
    logging.info('dataset {item}, element dict: {d}'.format(item=it, d=Counter(sp_lst)))

### load exp data and shuffle

test_structures = []
test_targets = []

data_path = 'data/all_data.json' # put here the path to the json file
with open(data_path,'r') as fp:
    d = json.load(fp)

s_exp = [Structure.from_dict(x['structure']) for x in d['ordered_exp'].values()]
t_exp = [x['band_gap'] for x in d['ordered_exp'].values()]

s_exp_disordered = [Structure.from_dict(x['structure']) for x in d['disordered_exp'].values()]
t_exp_disordered = [x['band_gap'] for x in d['disordered_exp'].values()]

# give a default but only single-fidelity
for i in range(len(s_exp)):
    s_exp[i].remove_oxidation_states()
    s_exp[i].state=[0]

for i in range(len(s_exp_disordered)):
    s_exp_disordered[i].remove_oxidation_states()
    s_exp_disordered[i].state=[0]

logging.info('exp data size is: {s}'.format(s=len(s_exp)))
r = list(range(len(list(d['ordered_exp'].keys()))))
random.shuffle(r)
sp_lst=[]
structures['E1'] = []
targets['E1'] = []
for i in r:
    sp_lst.extend(list(set(s_exp[i].species)))
    if random.random() > 0.5:
        structures['E1'].append(s_exp[i])
        targets['E1'].append(t_exp[i])
    else:
        test_structures.append(s_exp[i])
        test_targets.append(t_exp[i])

if swap_E1_test:
    structures['E1'], test_structures = test_structures, structures['E1']
    targets['E1'], test_targets = test_targets, targets['E1']

logging.info('dataset EXP, element dict: {d}'.format(item=it, d=Counter(sp_lst)))

logging.info(str(structures.keys()) + str(targets.keys()))
for k in structures.keys():
    logging.info(str(len(structures[k])) + str(len(targets[k])))

# data preprocess part
if load_old_model_enable:
    import pickle
    # load the past if needed
    model = MEGNetModel.from_file(old_model_name)
    if predict_before_dataclean:
        prediction(model, test_structures, test_targets)
    diff_lst = []
    for i in range(len(s_exp)):
        diff_lst.append(model.predict_structure(s_exp[i]).ravel() - t_exp[i])
    logging.info('Std of the list(model output - exp data) is: {std}, \
mean is: {mean}'.format(std=np.std(diff_lst),
                mean=np.mean(diff_lst)))

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
            # targets[i] = (model.predict_structure(structures[i]).ravel() + targets[i])/2
        logging.info('Data count: {dc}, std orig dft value: {std_orig}, std of model output: {std_model}'.format(
            dc=len(targets_lst), std_orig=np.std(targets_lst), std_model=np.std(prediction_lst)))
        logging.info('Data count: {dc}, Mean orig: {mean_orig}, Mean_model: {mean_model}'.format(
            dc=len(targets_lst), mean_orig=np.mean(targets_lst), mean_model=np.mean(prediction_lst)))
        f = open(dump_model_name + '_'+ it + '.txt', 'wb') # to store and analyze the error
        pickle.dump(error_lst, f)
        f.close()

# ordered structures only test
# model = MEGNetModel(nfeat_edge=10, nfeat_global=2, graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))

# ordered/disordered structures test together
model = MEGNetModel(nfeat_edge=100, nfeat_node=16, ngvocal=1, global_embedding_dim=16, graph_converter=CrystalGraphDisordered(bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5)))

model.save_model(dump_model_name+'_init_randomly' + '.hdf5')
init_model_tag = 'EGPHS'
start_model_tag = 'EGPHS'

ep = 5000
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

db_short_full_dict = {'G': 'gllb-sc', 'H': 'hse', 'S': 'scan', 'P': 'pbe', 'E': 'E1'}

def construct_dataset_from_str(db_short_str):
    s = []
    t = []
    for i in range(len(db_short_str)):
        s.extend(structures[db_short_full_dict[db_short_str[i]]])
        t.extend(targets[db_short_full_dict[db_short_str[i]]])
    if contain_e1_in_every_node:
        s.extend(structures['E1'])
        t.extend(targets['E1'])
    c = list(zip(s, t))
    random.shuffle(c)
    s, t = zip(*c)
    return s, t

def find_sub_tree(cur_tag, history_tag):
    global trained_last_time
    if init_model_tag == start_model_tag or trained_last_time == False:
        trained_last_time = False
        ###### load model #######
        father_model_name = dump_model_name + '_' + history_tag + '.hdf5'
        history_tag += '_'
        history_tag += cur_tag
        if special_path != '' and history_tag not in special_path:
            return
        else:
            pass

        if contain_e1_in_every_node:
            history_tag += 'E1'
        cur_model_name = dump_model_name + '_' + history_tag + '.hdf5'
        cur_model = MEGNetModel.from_file(father_model_name)
        ###### get dataset ######
        s, t = construct_dataset_from_str(cur_tag)
        l = len(s)
        ###### train ############
        try:
            cur_model.train(s[:int(0.8*l)], t[:int(0.8*l)],
                        validation_structures=s[int(0.8*l):],
                        validation_targets=t[int(0.8*l):],
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
        del s, t, l
        gc.collect()
    else:
        logging.info('cur_tag is {ct}, trained_last_time is {e}'.format(
            ct=str(cur_tag), e=str(trained_last_time)))
    ###### next level #######
    if len(cur_tag) > 1:
        for i in range(len(cur_tag)):
            next_tag = cur_tag[:i] + cur_tag[i+1:]
            find_sub_tree(next_tag, history_tag)
    elif contain_e1_in_every_node and trained_last_time == False:
    ####### extra E1 training ##
        s, t = construct_dataset_from_str('')
        l = len(s)
        try:
            cur_model.train(s[:int(0.8*l)], t[:int(0.8*l)],
                    validation_structures=s[int(0.8*l):],
                    validation_targets=t[int(0.8*l):],
                    callbacks=[callback],
                    save_checkpoint=False,
                    automatic_correction=False,
                    batch_size = 256,
                    epochs=ep)
        except TypeError:
            logging.info('MAE of {h}_E1 is: {mae}'.format(h=history_tag, mae='nan'))
        else:
            mae = prediction(cur_model, test_structures, test_targets)
            logging.info('Ordered structures MAE of {tag} is: {mae}'.format(tag=history_tag, mae=mae))
            mae = prediction(cur_model, s_exp_disordered, t_exp_disordered)
            logging.info('Disordered structures MAE of {tag} is: {mae}'.format(tag=history_tag, mae=mae))
        cur_model.save_model(dump_model_name + '_' + history_tag + '_E1.hdf5')
    else:
        pass
        
pbe_energy = prediction(model, structures['pbe'], targets['pbe'])
ordered_energy = prediction(model, test_structures, test_targets)
disordered_energy = prediction(model, s_exp_disordered, t_exp_disordered)

logging.info('Prediction before trainnig, MAE of \
        pbe: {pbe}; ordered: {ordered}; disordered: {disordered}.'.format(
    pbe=pbe_energy, ordered=ordered_energy, disordered=disordered_energy))

find_sub_tree(init_model_tag, 'init_randomly')

