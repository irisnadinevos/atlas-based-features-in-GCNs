# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:42:36 2021

@author: iris

"""

import torch
import numpy as np
import networkx as nx
import pandas as pd

from torch_geometric.data import InMemoryDataset, Data

import src.config as config
from src.dataset.embedding import getDirEmb, readTxt

def get_dataset(root, features, reset=False, transform=None, pre_transform=None):
    
    if reset:
        files = list(root.glob('processed/*'))
        for f in files:
            f.unlink()
    
    dataset = GraphDataset(root, features, transform, pre_transform)
    
    return dataset

def NodePositions_NormalizeScale(data): # adapted from torch_geometric.transforms.normalizescale

    for store in data.node_stores:
        
        if hasattr(store, 'pos'):
            store.pos = store.pos - store.pos.mean(dim=-2, keepdim=True)
            
    scale = (1 / data.pos.abs().max()) * 0.999999
    data.pos = data.pos * scale
    
    return data


class GraphDataset(InMemoryDataset):
    
    def __init__(self, root, features, transform=None, pre_transform=None):
        
        self.features = features
        
        super(GraphDataset, self).__init__(root, features, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        features = self.features
                
        filenames = list(self.root.glob('*.gexf'))
        
        data_list = []

        for indFile, filename in enumerate(filenames):
            
            G = nx.read_gexf(filename)
            mapping = {name: j for j, name in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            G.remove_edges_from(nx.selfloop_edges(G))
            
            data_x = np.zeros((nx.number_of_nodes(G), len(features)))
            
            d = {f:[] for f in features}
            
            node_pos = nx.get_node_attributes(G,'pos')
            node_pos_list = torch.FloatTensor([list(i[0]) for i in node_pos.values()])
            data_node_pos = Data(pos=node_pos_list)
            normalized_pos = NodePositions_NormalizeScale(data_node_pos)
            
            #% single extra marker position to coordinate system
            transformix_file = config.TRANSFORMIX / filename.stem / 'outputpoints.txt'
                                         
            coord = readTxt(transformix_file)
            extra_node_pos = torch.cat((node_pos_list, torch.tensor([coord])))
            extra_data_node_pos = Data(pos=extra_node_pos)
            extra_normalized_pos = NodePositions_NormalizeScale(extra_data_node_pos)

            for ind, nodes_feature in enumerate(features):
                
                if nodes_feature == 'nodePos':
                    d[nodes_feature].extend(normalized_pos.pos)
                    
                
                elif nodes_feature == 'totalEL':
                    edgeLen_dict = nx.get_node_attributes(G, nodes_feature)
                    data_edgeLen = [np.log(x/100) for x in edgeLen_dict.values()]
                    d[nodes_feature].extend(data_edgeLen)
                    
                
                elif 'atlas' in nodes_feature:
                    atlas_dict = nx.get_node_attributes(G, nodes_feature)
                    data_atlas = [1/x for x in atlas_dict.values()]
                    d[nodes_feature].extend(data_atlas)
                    
                    
                elif nodes_feature == 'dirEmb':
                    dirEmb_array = np.zeros((len(normalized_pos.pos), 25))
                    for i, node_pos in enumerate(extra_normalized_pos.pos[:-1]):
                        dirVector = node_pos-extra_normalized_pos.pos[-1]
                        dirVectorNorm = dirVector/np.linalg.norm(dirVector)
                        embedding, (theta, phi) = getDirEmb(dirVectorNorm)
                        dirEmb_array[i] = embedding
                        
                    d[nodes_feature].extend(dirEmb_array)#, axis=1)   
                    
                    
                else: # nodes_feature == rad or nodes_feature == numEdges
                    feat_dict = nx.get_node_attributes(G, nodes_feature)
                    data_feat = [x for x in feat_dict.values()]
                    d[nodes_feature].extend(data_feat)

            df = pd.DataFrame(d)
            data_x = np.array([np.hstack(x) for x in df.values])
  
            labels_dict = nx.get_node_attributes(G, 'node') #ground truth label class
            data_y = np.array([x for x in labels_dict.values()])
            
            
            get_edges = np.asarray(list(G.edges))
            undirected_graph_edges = [[int(i), int(j)] for i,j in get_edges] + [[int(j), int(i)] for i,j in get_edges]
            edge_index = torch.LongTensor(undirected_graph_edges)
            
            pos_dict = nx.get_node_attributes(G,'pos')
            data_pos = torch.FloatTensor([list(i[0]) for i in pos_dict.values()])
            
            data = Data(x=torch.FloatTensor(data_x), edge_index=edge_index.t().contiguous(), edge_attr=None, y=torch.LongTensor(data_y), pos=data_pos)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])        
        
    def __getitem__(self, idx):
       	if isinstance(idx, int):
       		data = self.get(self.indices()[idx])
       		#data = data if self.transform is None else self.transform(data)
       		return data
       	else:
       		return self.index_select(idx)