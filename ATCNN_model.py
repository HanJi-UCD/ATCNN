#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 20:59:37 2022

@author: han
"""

import torch
import torch.nn as nn
import numpy as np
import math
import copy
from copy import deepcopy

# case of 4 LiFi size
class ATCNN(nn.Module):
    def __init__(self, net='resnet18', input_dim = 6, cond_dim = 300, cond_out_dim = 6, output_dim = 5):
        super().__init__()
        
        self.condition = nn.Sequential(nn.Linear(cond_dim, 64, bias=True), #1
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(inplace=True), # first layer
                                        
                                        # nn.Linear(128, 32, bias=True), #2
                                        # nn.BatchNorm1d(32),
                                        # nn.ReLU(inplace=True), # second layer
                                        
                                        nn.Linear(64, cond_out_dim, bias=True), #3
                                        nn.BatchNorm1d(cond_out_dim),
                                        nn.ReLU(inplace=True), # third layer
                                        )
        self.target = nn.Linear(input_dim, input_dim, bias=False)
        self.combiner = nn.Sequential(nn.Linear(cond_out_dim+input_dim, output_dim, bias=False),
                                        nn.Softmax(dim=1)
                                        ) 
    # input: target (6 dim) and condition (300 dim), 
    # output: result (5 dim)  
    
    def forward(self, ipt, cond):
        ipt = self.target(ipt)        
        cond = self.condition(cond)
        x = torch.cat((ipt, cond), dim=1)
        x = self.combiner(x)
        return x

class ATCNN_9LiFi(nn.Module):
    def __init__(self, net='resnet18', input_dim = 6, cond_dim = 300, cond_out_dim = 6, output_dim = 5):
        super().__init__()
        
        self.condition = nn.Sequential(nn.Linear(cond_dim, 128, bias=True), #1
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True), # first layer
                                        
                                        nn.Linear(128, 32, bias=True), #2
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(inplace=True), # second layer
                                        
                                        nn.Linear(32, cond_out_dim, bias=True), #3
                                        nn.BatchNorm1d(cond_out_dim),
                                        nn.ReLU(inplace=True), # third layer
                                        )
        self.target = nn.Linear(input_dim, input_dim, bias=False)
        self.combiner = nn.Sequential(nn.Linear(cond_out_dim+input_dim, output_dim, bias=False),
                                        nn.Softmax(dim=1)
                                        ) 
    # input: target (6 dim) and condition (300 dim), 
    # output: result (5 dim)  
    
    def forward(self, ipt, cond):
        ipt = self.target(ipt)        
        cond = self.condition(cond)
        x = torch.cat((ipt, cond), dim=1)
        x = self.combiner(x)
        return x

class global_dnn(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, net='resnet18', 
                 input_dim = 300, 
                 output_dim = 250,
                 ):

        super(global_dnn, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(input_dim, 128, bias=True), #1
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True), # first layer
                                        
                                        nn.Linear(128, 64, bias=True),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(inplace=True), # fourth layer
 
                                        nn.Linear(64, output_dim, bias=True),
                                        nn.BatchNorm1d(output_dim),
                                        nn.Sigmoid(), # sixth layer
                                        )

    def forward(self, x):
        x = self.net(x)
        return x

class global_dnn_9LiFi(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, net='resnet18', 
                 input_dim = 550, 
                 output_dim = 500,
                 ):

        super(global_dnn_9LiFi, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(input_dim, 256, bias=True), #1
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=True), # first layer
                                        
                                        nn.Linear(256, 64, bias=True),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(inplace=True), # fourth layer
                                        
                                        nn.Linear(64, 128, bias=True),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True), # fourth layer
 
                                        nn.Linear(128, output_dim, bias=True),
                                        nn.BatchNorm1d(output_dim),
                                        nn.Sigmoid(), # sixth layer
                                        )

    def forward(self, x):
        x = self.net(x)
        return x
    
def mirror(M, AP_num, UE_num, condition):
    # M is the maximum UE number of ML model
    quotient = math.floor(M/UE_num)
    reminder = M % UE_num
    mirroring_condiiton = []
    new_cond = np.array(condition)
    if reminder == 0: # split R equally
        for i in range(UE_num):
            new_cond[(AP_num+1)*i + AP_num] = new_cond[(AP_num+1)*i + AP_num]/quotient
    else: # not split R equally
        for i in range(reminder):
            new_cond[(AP_num+1)*i + AP_num] = new_cond[(AP_num+1)*i + AP_num]/(quotient+1)
        for i in range(UE_num - reminder):
            new_cond[(AP_num+1)*(i+reminder) + AP_num] = new_cond[(AP_num+1)*(i+reminder) + AP_num]/quotient   
    new_cond = list(new_cond)
    # extend length from UE_num to M
    if quotient == 1: # UE number is bigger than 25
        mirroring_condiiton.append(new_cond)
        mirroring_condiiton.append(new_cond[0 : (AP_num+1)*reminder])
    else:
        for i in range(quotient):
            mirroring_condiiton.append(new_cond)
        mirroring_condiiton.append(new_cond[0: (AP_num+1)*reminder])
    mirroring_condiiton = sum(mirroring_condiiton, []) # flatten the list
    return mirroring_condiiton

def to_binary(AP_num, vector):
    vector = vector[0]
    binary_vector = []
    for i in range(AP_num):
        if vector[i] == max(vector):
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    return binary_vector

def switch(ipt, idx, user_dim, mode=None):   
    if mode == None: #swtich the user data on idx to the first position
        temp = deepcopy(ipt[..., idx*user_dim:idx*user_dim+user_dim])
        ipt[..., idx*user_dim:idx*user_dim+user_dim] = deepcopy(ipt[..., 0:user_dim])
        ipt[..., 0:user_dim] = temp
    else: # switch idx UE into the last position
        temp = deepcopy(ipt[..., idx*user_dim:idx*user_dim+user_dim])
        ipt[..., idx*user_dim:idx*user_dim+user_dim] = deepcopy(ipt[..., -user_dim:])
        ipt[..., -user_dim:] = temp
        
    return ipt.tolist()

# output X_iu for each UE
def translate(binray_output):
    X_iu = []
    for i in range(len(binray_output)): 
        AP_index = binray_output[i].index(1) + 1
        X_iu.append(AP_index)
    return X_iu





















