# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:16:28 2023

@author: iris
"""

import yaml


class parameters(object):
    
    def __init__(self, data):
        
        for key in data:
            setattr(self, key, data[key])
              

def read_par_file(config_file):
    
    with open(config_file) as file:
        par = yaml.load(file, Loader=yaml.FullLoader)
    
    return parameters(par)
