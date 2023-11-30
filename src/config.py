# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:15:21 2023

@author: iris
"""

from pathlib import Path



PROJECT_DIR = Path(__file__).parent.absolute().parent.absolute()

ROOT_TRAIN    = ''
ROOT_TEST     = ''
ROOT_ALL      = ''

RESULTS_TRAIN = Path('./results/output/train')
RESULTS_TEST  = Path('./results/output/test')

PAR_FILES     = Path('./par_files')
TRANSFORMIX   = ''

LEARNING_RATE   = 0.001
BATCH_SIZE      = 25
NUM_IT          = 5000
SEED            = 0
N_KFOLD         = 10
WEIGHTS         = [50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
PATIENCE        = 10

KEY_EXP_NAME    = Path('./dataset')


MODE_TRAIN_CV   = 0
MODE_VALID_CV   = 0
MODE_TRAIN_FULL = 0
MODE_TEST       = 1

ACT_DROPOUT     = False
DROPOUT_RATE    = 0

ACT_POSTP       = True

MC_DROPOUT      = False