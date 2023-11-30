# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:53:58 2023

@author: iris
"""

import json
import numpy as np

from netcal.metrics import ECE

import src.config as config
    
def read_json(path):
    
    with open(path) as f:
        data = json.load(f)
    f.close()
    
    return data

def save_output(preds, result_path_test):
    
    files = list(config.ROOT_TEST.glob('*.gexf'))
    filenames = [f.stem for f in files]
    
    for i, p in enumerate(preds):
        graph_name = filenames[i]
        with open(result_path_test / f'Pred_{graph_name}.json', 'w', encoding ='utf8') as json_file:
           json.dump(p, json_file, indent=4)
           
    return

def print_metrics(path):
    
    n_bins = 10
    ece_score = ECE(n_bins, detection=False)
    
    if config.MODE_VALID_CV:
        
        TPs = np.array(read_json(path / 'TPs.json'))[:, :, 1:]
        FPs = np.array(read_json(path / 'FPs.json'))[:, :, 1:]
        FNs = np.array(read_json(path / 'FNs.json'))[:, :, 1:]
        
        TPs_sum = np.sum(np.sum(TPs, axis=0), axis=1) # [fold, graph, class]
        FNs_sum = np.sum(np.sum(FNs, axis=0), axis=1)
        FPs_sum = np.sum(np.sum(FPs, axis=0), axis=1)
        

    else:
        
        TPs = np.array(read_json(path / 'TPs.json'))[:, 1:]
        FPs = np.array(read_json(path / 'FPs.json'))[:, 1:]
        FNs = np.array(read_json(path / 'FNs.json'))[:, 1:]
        
        TPs_sum = np.sum(TPs, axis=1) # [graph, class]
        FNs_sum = np.sum(FNs, axis=1)
        FPs_sum = np.sum(FPs, axis=1)

        confs = np.array(read_json(path / 'confs.json'))
        fracs = np.array(read_json(path / 'fracs.json'))
        
        eces = [ece_score.measure(np.array(confs[i]), np.array(fracs[i])) for i in range(len(confs))]
        
        print(f'ECE: {np.round(np.mean(eces), 4)} +/- {np.round(np.std(eces), 4)}')
    
    recall = np.array([TPs_sum[i]/(TPs_sum[i] + FNs_sum[i]) for i in range(len(TPs_sum))])
    precision = np.array([TPs_sum[i]/(TPs_sum[i] + FPs_sum[i]) for i in range(len(TPs_sum))])
    
    print(path.stem)
    print(f'Recall: {np.round(np.mean(recall), 2)} +/- {np.round(np.std(recall), 2)}')
    print(f'Precision: {np.round(np.mean(precision), 2)} +/- {np.round(np.std(precision), 2)}')
    print(f'FPs: {np.round(np.mean(FPs_sum), 2)} +/- {np.round(np.std(FPs_sum), 2)}')
    print(f'FNs: {np.round(np.mean(FNs_sum), 2)} +/- {np.round(np.std(FNs_sum), 2)}')

    return recall, precision
