# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 20:36:01 2021

@author: iris
"""

import json
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from src.model import models
import src.config as config
from src.evaluation.metrics import true_positive, false_positive, false_negative, true_negative

torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def do_test(dataset, conv_operator, model_path, result_path):
            
    test_loader = DataLoader(dataset)

    model = getattr(models, conv_operator)(dataset).to(device)
    model.load_state_dict(torch.load(model_path))
    
    d_metrics = {'TPs': torch.zeros(len(dataset), dataset.num_classes),\
                 'FPs': torch.zeros(len(dataset), dataset.num_classes),\
                 'FNs': torch.zeros(len(dataset), dataset.num_classes),\
                 'TNs': torch.zeros(len(dataset), dataset.num_classes)}

    preds = []

    all_confidences = []
    all_targets = []
    
    
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            target = data.y
            out = model(data.x, data.edge_index)
            
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
                        
            d_metrics['TPs'][i][range(dataset.num_classes)] = true_positive(pred, target, dataset.num_classes).float()
            d_metrics['FPs'][i][range(dataset.num_classes)] = false_positive(pred, target, dataset.num_classes).float()
            d_metrics['FNs'][i][range(dataset.num_classes)] = false_negative(pred, target, dataset.num_classes).float()
            d_metrics['TNs'][i][range(dataset.num_classes)] = true_negative(pred, target, dataset.num_classes).float()
            
            # Uncertainty measures
            all_confidences.append(torch.exp(out).cpu())
            all_targets.append(target.cpu())
            
            preds.append(pred.tolist())
            
    for metric in d_metrics.keys(): 
        with open(result_path / f'{metric}.json', 'w', encoding ='utf8') as json_file:
            json.dump(d_metrics[metric].tolist(), json_file, indent=4)
    
    with open(result_path / 'confs.json', 'w', encoding ='utf8') as json_file:
        json.dump([c.tolist() for c in all_confidences], json_file, indent=4)

    with open(result_path / 'fracs.json', 'w', encoding ='utf8') as json_file:
        json.dump([t.tolist() for t in all_targets], json_file, indent=4)
        
    return preds


def do_MCdropout(dataset, conv_operator, model_path):
    
    test_loader = DataLoader(dataset)

    model = getattr(models, conv_operator)(dataset)
    model.load_state_dict(torch.load(model_path))
    
    model.eval()
    
    mean_variation = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            out = model(data)
                             
            out_mcd = torch.zeros((100, data.num_nodes, dataset.num_classes))
            
            for i in range(100):  
        
                with torch.no_grad():
                    out = model(data)
                    out_mcd[i, ...] = torch.exp(out)
            
            mean_variation.append(np.mean(np.var(np.array(out_mcd), axis=0), axis=0))