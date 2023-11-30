# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:04:24 2022

@author: iris
"""

import numpy as np
from math import pi as PI
import re

def readTxt(file, points='Single'):
    # Output coordinates = Registered atlas points on patient TOF
    coord = []
    
    with open(file, 'r') as fp:
      for line in fp:
        match = re.search('OutputPoint = ([^,;]*)', line)
        point = re.search('Point	(\d+)', line)
        
        if points == 'Single':
            if match and point.group(1) == '2':
              outString = match.group(1)
              findDecimals = re.findall('[-+]?\d*\.?\d+', outString)
              
              coord.extend([float(x) for x in findDecimals])
        else:
            if match:
              outString = match.group(1)
              findDecimals = re.findall('[-+]?\d*\.?\d+', outString)
              
              coord.append([float(x) for x in findDecimals])
              
    return coord


def getDirEmb(unit_vector):
    
    theta_ranges = []
    i = 0
    while i<360:
        theta_ranges.append((i,i+72))
        i+= 72
    phi_ranges = []
    i = 0
    while i<180:
      phi_ranges.append((i,i+36))
      i+=36
    
    all_ranges = []
    for rangeTheta in theta_ranges:
      for rangePhi in phi_ranges:
        all_ranges.append([rangeTheta, rangePhi])
      
    xy = unit_vector[0]**2 + unit_vector[1]**2
    phi = np.arctan2(np.sqrt(xy), unit_vector[2])* 180 / PI
    theta = np.arctan2(unit_vector[1], unit_vector[0])* 180 / PI
    
    if theta < 0:
      theta += 360

    edgeArray = [lowerT <= theta <= upperT and lowerP <= phi <= upperP for (lowerT,upperT),(lowerP,upperP) in all_ranges]
    edgeEmb = np.array(edgeArray).astype(int)
  
    return edgeEmb, (phi, theta)