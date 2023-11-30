# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:50:35 2023

@author: iris
"""

import json
import numpy as np
import multiprocessing as mp

import src.config as config
import src.model.trainer as trainer
import src.model.tester as tester
from src.dataset.dataset import get_dataset
from src.dataset.utils import read_par_file
from src.evaluation.utils import read_json, print_metrics, save_output


def run(c):
    
    features = read_par_file(c).features
    conv_operator = read_par_file(c).net
    
    result_path_cv = config.RESULTS_TRAIN / 'cv' / config.KEY_EXP_NAME / c.stem
    result_path_train = config.RESULTS_TRAIN / 'full' / c.stem
    result_path_test = config.RESULTS_TEST / c.stem
 
    
    if config.MODE_TRAIN_CV:
        train_dataset = get_dataset(config.ROOT_TRAIN, features, reset=True)
        
        result_path_cv.mkdir(parents=True)

        with open(result_path_cv / 'params.json', 'w', encoding ='utf8') as json_file:
            json.dump([config.SEED, features, config.ACT_DROPOUT, config.DROPOUT_RATE, config.ACT_POSTP], json_file, indent=4)

        trainer.do_training(train_dataset, conv_operator, result_path_cv, stop_it=None)
  
    
    if config.MODE_TRAIN_FULL:
        train_dataset = get_dataset(config.ROOT_TRAIN, features, reset=True)
        es_data = read_json(result_path_cv / 'early_stopping.json')
        stop_it = int(np.mean(es_data) + np.mean(es_data)*0.2)

        trainer.do_training(train_dataset, conv_operator, result_path_train, stop_it=stop_it)
        
    
    if config.MODE_TEST:
        
        if not result_path_test.exists():
            result_path_test.mkdir()
            test_dataset = get_dataset(config.ROOT_TEST, features, reset=True)
            preds = tester.do_test(test_dataset, conv_operator, model_path=f'{result_path_train}_checkpoint.pt', result_path=result_path_test)
            save_output(preds, result_path_test)
        

if __name__ == '__main__':
    
    par_files = list(config.PAR_FILES.glob('*'))
    
    if config.MODE_TRAIN_CV or config.MODE_TRAIN_FULL:
        
        with mp.Pool(processes=4) as pool:
            pool.map(run, par_files)
        pool.close()
        

    if config.MODE_VALID_CV:
        # show results cross validation
        recall = []
        precision = []
        stems = []
        for c in par_files:
            r, p = print_metrics(config.RESULTS_TRAIN / 'cv' / config.KEY_EXP_NAME / c.stem)
            recall.append(r)
            precision.append(p)
            stems.append(c.stem)
            print(read_json(config.RESULTS_TRAIN / 'cv' / config.KEY_EXP_NAME / c.stem / 'params.json'))
            print('\n')
        
        
    if config.MODE_TEST:  
        # show results on test set for two models (including calibration)
        
        best_model_1 = config.PAR_FILES / 'GCNNet.yml'
        best_model_2 = config.PAR_FILES / 'GraphNet_atlasdistdir.yml'
        
        data_1 = run(best_model_1)
        data_2 = run(best_model_2)
    