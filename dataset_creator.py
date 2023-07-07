#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:21:35 2022

@author: qiang
"""
#%%

import numpy as np
import csv
from glob import glob
import torch
import os

# Mannully set the path to the folder where storing the data
mode = 'input'
dataset_path = 'dataset/GT_9LiFi_3W_DNN/15UE/'

if mode == 'input':
    raw_dataset_path = dataset_path+'input_test'
    output_file_name = dataset_path+'input_test.h5'
elif mode == 'output':
    raw_dataset_path =  dataset_path+'output_test'
    output_file_name =  dataset_path+'output_test.h5'
    
paths = glob(raw_dataset_path + '/*')

for idx in range(len(paths)): 
    with open(os.path.join(raw_dataset_path, f'nor_input_batch{idx+1001}.csv'), encoding="utf-8") as f:
        reader = csv.reader(f)
        sub_dataset = []
        for row in reader:
            data_row = []
            for fig in row:
                data_row.append(float(fig.strip()))
            sub_dataset.append(data_row)
        sub_dataset = np.array(sub_dataset).transpose(1,0)

        if idx == 0:
            dataset = sub_dataset
        else:
            dataset = np.concatenate((dataset, sub_dataset), axis=0)
        if idx % 100 == 0:
            print(f'{idx}/{len(paths)} sets have been finished')

print(f'Saving to file {output_file_name}')
print('Shape of the dataset is: ', dataset.shape)
    
torch.save(dataset, output_file_name)

#%%
import numpy as np
import csv
from glob import glob
import torch
import os

mode = 'input'
dataset_path = 'dataset/GT_9LiFi_3W_DNN/50UE/'

if mode == 'input':
    raw_dataset_path = dataset_path+'input_train'
    output_file_name = dataset_path+'input_train.h5'
elif mode == 'output':
    raw_dataset_path =  dataset_path+'output_train'
    output_file_name =  dataset_path+'output_train.h5'
    
paths = glob(raw_dataset_path + '/*')

for idx in range(len(paths)): 
    with open(os.path.join(raw_dataset_path, f'nor_input_batch{idx+901}.csv'), encoding="utf-8") as f:
        reader = csv.reader(f)
        sub_dataset = []
        for row in reader:
            data_row = []
            for fig in row:
                data_row.append(float(fig.strip()))
            sub_dataset.append(data_row)
        sub_dataset = np.array(sub_dataset).transpose(1,0)
        if idx == 0:
            dataset = sub_dataset
        else:
            dataset = np.concatenate((dataset,sub_dataset),axis=0)
        if idx % 100 == 0:
            print(f'{idx}/{len(paths)} sets have been finished')

print(f'Saving to file {output_file_name}')
print('Shape of the dataset is: ', dataset.shape)
    
torch.save(dataset,output_file_name)

