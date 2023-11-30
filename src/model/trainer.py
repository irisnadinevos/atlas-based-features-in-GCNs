# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:51:15 2023

@author: iris
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score

import src.config as config
import src.model.models as models

from src.model.utils import EarlyStopping
from src.model.models import getClassWeights, resetModelWeights
from src.evaluation.metrics import true_positive, false_positive, false_negative, true_negative


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

    return
    

def plot_losses(train_loss, valid_loss=None):

    plt.figure()     
    plt.plot(train_loss)
    legend = ['Train']
    if valid_loss:
        plt.plot(valid_loss)
        legend.append('Validation')
    plt.legend(legend)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()       
    
    return

def train_one_fold(dataset, conv_operator, fold, train_idx, valid_idx, result_path):
    
    weightArray = getClassWeights(dataset, dataset.num_classes)
    lossF = torch.nn.NLLLoss(weight=torch.tensor(weightArray).to(device))
    model = getattr(models, conv_operator)(dataset).to(device)
    model.apply(resetModelWeights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=5e-4)
    
    train_sampler =  dataset[list(train_idx)]
    valid_sampler =  dataset[list(valid_idx)]
    
    train_loader = DataLoader(train_sampler, batch_size=config.BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(valid_sampler)
    
    train_loss = []
    valid_loss = []
    
    # Training
    early_stopping = EarlyStopping(patience=config.PATIENCE, verbose=False, path=result_path / f"checkpoint_{fold}.pt")
    
    data_iter = iter(train_loader)
    
    for i in range(config.NUM_IT):
        
        try:
            data = next(data_iter)
            
        except StopIteration:
            train_loader = DataLoader(train_sampler, batch_size=config.BATCH_SIZE, shuffle=False)
            data_iter = iter(train_loader)
            data = next(data_iter)

        model.train()
        
        data = data.to(device)
        target = data.y
        
        optimizer.zero_grad()       
        out = model(data.x, data.edge_index)
        loss = lossF(out, target)       
        loss.backward()
        optimizer.step()
                   
        train_loss.append(loss.item())#/len(train_loader))
   
        if i % 500 == 0:
            print(f'Train it: {i}/{config.NUM_IT}, Loss: {loss.item()}')

        _, val_loss = do_validation(model, dataset.num_classes, valid_loader, lossF, model_checkpoint=False)
        valid_loss.append(val_loss)
        
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            early_stopping_it.append(i)
            
            print(f"Early stopping at iteration {i}.")
            break

    model.load_state_dict(torch.load(result_path / f"checkpoint_{fold}.pt"))
    do_validation(model, dataset.num_classes, valid_loader, lossF, model_checkpoint=True)
    
    plot_losses(train_loss, valid_loss)
    
    return
    
    
def train_all(dataset, conv_operator, result_path, stop_it):

    weightArray = getClassWeights(dataset, dataset.num_classes)
    lossF = torch.nn.NLLLoss(weight=torch.tensor(weightArray).to(device))
    model = getattr(models, conv_operator)(dataset).to(device)
    model.apply(resetModelWeights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=5e-4)
    
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Training
    train_loss = []
    
    data_iter = iter(train_loader)
        
    for i in range(stop_it):
        
        try:
            data = next(data_iter)
            
        except StopIteration:
            train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            data_iter = iter(train_loader)
            data = next(data_iter)

        model.train()
        
        data = data.to(device)
        target = data.y
        
        optimizer.zero_grad()       
        out = model(data.x, data.edge_index)
        loss = lossF(out, target)       
        loss.backward()
        optimizer.step()
            
        train_loss.append(loss.item())
       
        if i % 500 == 0:
            print(f'Train it: {i}/{stop_it}, Loss: {loss.item()}')

    print(f'Training stopped at it {i}')  
    torch.save(model.state_dict(), f'{result_path}_checkpoint.pt')
    
    plot_losses(train_loss)
    
    return

def do_validation(model, num_classes, valid_loader, lossF, model_checkpoint=False):
    
    running_loss = 0.0
    
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            data = data.to(device)
            target = data.y      
            out = model(data)
            loss = lossF(out, target)
            running_loss += loss.item()  
            
            if model_checkpoint:
                _, pred = torch.exp(out).max(dim=1)
                
                if config.ACT_POSTP:
                    for ii in range(1, 13):
                        if len(np.where(pred.cpu() == ii)[0]) > 1:
                            highest_pred = -10
                            for indp, p in enumerate(np.where(pred.cpu() == ii)[0]):
                                if out[p][ii].item() > highest_pred:
                                    highest_pred = out[p][ii].item()
                                    highest_ind = p
                            for pp in np.where(pred.cpu() == ii)[0]:
                                if pp != highest_ind:
                                    pred[pp] = 0
                                
                d_metrics['TPs'][fold][i][range(num_classes)] = true_positive(pred, target, num_classes).float()
                d_metrics['FPs'][fold][i][range(num_classes)] = false_positive(pred, target, num_classes).float()
                d_metrics['FNs'][fold][i][range(num_classes)] = false_negative(pred, target, num_classes).float()
                d_metrics['TNs'][fold][i][range(num_classes)] = true_negative(pred, target, num_classes).float()
                
        valid_loss = running_loss/len(valid_loader)
    
    return d_metrics, valid_loss

    
def do_training(dataset, conv_operator, result_path, stop_it=None):
    
    global d_metrics
    global fold
    global early_stopping_it
    
    d_metrics = {'TPs': torch.zeros(config.N_KFOLD, int(len(dataset)/config.N_KFOLD), dataset.num_classes),\
                     'FPs': torch.zeros(config.N_KFOLD, int(len(dataset)/config.N_KFOLD), dataset.num_classes),\
                     'FNs': torch.zeros(config.N_KFOLD, int(len(dataset)/config.N_KFOLD), dataset.num_classes),\
                     'TNs': torch.zeros(config.N_KFOLD, int(len(dataset)/config.N_KFOLD), dataset.num_classes)}
    
    early_stopping_it = []
    
    torch.manual_seed(config.SEED)

    if config.MODE_TRAIN_CV:
        
        kf = KFold(n_splits=config.N_KFOLD, shuffle=False) 
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):  
            print('\nFold : {}'.format(fold))
                  
            train_one_fold(dataset, conv_operator, fold, train_idx, valid_idx, result_path)         
            
        for metric in d_metrics.keys(): 
            with open(result_path / f'{metric}.json', 'w', encoding ='utf8') as json_file:
                json.dump(d_metrics[metric].tolist(), json_file, indent=4)
                
        with open(result_path / 'early_stopping.json', 'w', encoding ='utf8') as json_file:
            json.dump(early_stopping_it, json_file, indent=4)
    else:
   
        train_all(dataset, conv_operator, result_path, stop_it)
        
    return