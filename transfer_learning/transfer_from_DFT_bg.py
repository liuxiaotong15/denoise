import random
import pandas as pd
import numpy as np
from pymatgen.core import Structure
import json

from maml.models import AtomSets

from pymatgen.core.structure import Structure

import tensorflow as tf
from megnet.data.crystal import CrystalGraph, CrystalGraphDisordered
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel
from maml.describers import MEGNetStructure
from maml.models import MLP
structures = {}
targets = {}
test_structures = []
test_targets = []
dft_structures = []
dft_targets = []

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

import logging
root_logger = logging.getLogger()
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename="tl_dft.log",
        format='%(asctime)s-%(pathname)s[line:%(lineno)d]-%(levelname)s: %(message)s',
        level=logging.INFO)

items = ['gllb-sc', 'pbe', 'scan', 'hse']
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
        tmp.state=[0]
        dft_structures.append(tmp)
        dft_targets.append(df[it+'_gap'][i])

c = list(zip(dft_structures, dft_targets))
random.shuffle(c)
dft_structures, dft_targets = zip(*c)
## load DFT data finished ##

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
            s_exp[i].state = [0]

        if fold_i  == test_idx:
            test_strcts = s_exp
            test_tgts = t_exp
        else:
            train_structures.extend(s_exp)
            train_targets.extend(t_exp)
    # no need to shuffle, we have shuffled in the splitting script
    return train_structures, train_targets, test_strcts, test_tgts



def main():
    ret = []
    global structures, targets, test_structures, test_targets

    model_megnet = MEGNetModel(nfeat_edge=100, nfeat_node=16, ngvocal=1, global_embedding_dim=16, graph_converter=CrystalGraphDisordered(bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5)))
    dft_len = len(dft_structures)
    model_megnet.train(dft_structures[:int(dft_len * 0.8)], 
            dft_targets[:int(dft_len * 0.8)],
            validation_structures=dft_structures[int(dft_len * 0.8):],
            validation_targets=dft_targets[int(dft_len * 0.8):],
            callbacks=[callback],
            save_checkpoint=False,
            automatic_correction=False,
            batch_size = 256,
            epochs=5000)
    model_megnet.save_model("dft_bandgap.hdf5")
    
    for i in range(5):
        logging.info('Fold-{0} start'.format(i))
        
        structures['E1'], targets['E1'], test_structures, test_targets = load_exp_data(i)
        logging.info('E1 size: {0}, test size: {1}'.format(len(structures['E1']), len(test_structures)))
        describer = MEGNetStructure(mode='final', feature_batch='pandas_concat', name="dft_bandgap.hdf5")

        model = MLP(describer=describer,
                   input_dim=96,
                   compile_metrics=['mae'],
                   loss='mae',
                   is_classification=False)
        
        features = describer.transform(structures['E1'])
        model.fit(features, targets['E1'], validation_split=0.8, epochs=1000, 
                callbacks=[callback])
        
        test_features = describer.transform(test_structures)
        loss, metric = model.evaluate(test_features, test_targets, True)
        
        logging.info(f"The MAE is {metric:.3f} eV/atom")
        ret.append(metric)
    logging.info(ret)
    logging.info(np.mean(ret))
        

if __name__ == "__main__":
    main()
