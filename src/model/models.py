# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:27:19 2021

@author: iris
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv

import src.config as config

class GCNNet(torch.nn.Module):
    
    def __init__(self, dataset): 
        super(GCNNet, self).__init__()
        
        self.conv1 = GCNConv(dataset.num_node_features, 64)
        self.conv2 = GCNConv(64, dataset.num_classes)
        

    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        if config.MC_DROPOUT:
            x = F.dropout(x, p=0.5, training=True)
        else:
            pass
        
        if config.ACT_DROPOUT:
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)
        else:
            pass
            
        
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    

class GraphNet(torch.nn.Module):
    
    def __init__(self, dataset): 
        
        super(GraphNet, self).__init__()
        self.conv1 = GraphConv(dataset.num_node_features, 64)
        self.conv2 = GraphConv(64, dataset.num_classes)

    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)    
        
        if config.MC_DROPOUT:
            x = F.dropout(x, p=0.5, training=True)
        else:
            pass
        
        if config.ACT_DROPOUT:
            x = F.dropout(x, p=config.DROPOUT_RATE, training=self.training)
        else:
            pass
            
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    

def resetModelWeights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  ''' 
  for layer in m.children():
      
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    
    layer.reset_parameters()
    
    
def getClassWeights(dataset, classes):
    
    weightArray = []
    numberOther = sum([(np.count_nonzero(data.y==0)) for data in dataset])
    weightArray.append(numberOther)
    
    s = 0
    
    for i in range(1, classes):
        occTargetNode = sum([(np.count_nonzero(data.y==i)) for data in dataset])
        s += occTargetNode
        weightArray.append(occTargetNode)
    
    weightArray = [1/x for x in weightArray]
    
    for i, w in enumerate(config.WEIGHTS):
        weightArray[i] *= w

    return weightArray