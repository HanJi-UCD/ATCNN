# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 21:37:25 2022

@author: Han
"""

import torch
import numpy as np
import pandas as pd
import warnings
import time
from utils import mapping, normalization, save_list_to_csv
from ATCNN_model import ATCNN, switch, to_binary
warnings.filterwarnings("ignore")

# test normalised and mirrored dataset
def test_acc_ATCNN(input_path, output_path, AP_num, UE_num):
    #print('******Loading Input data******')
    data = pd.read_csv(input_path, header=None)
    values = np.array(data).T.tolist() # with dimension of 256*300
    trained_output = []
    test_UE_num = 1 # revise here

    for i in range(256):
        condition = values[i]
        condition = torch.tensor(condition)
        output_instance = []
        for j in range(test_UE_num):
            raw_condition = torch.tensor(values[i])
            condition_now = switch(raw_condition, j, AP_num+1) 
            condition_now = torch.tensor([condition_now])
            
            Target = condition_now[..., 0:user_dim]
            Condition = condition_now[..., 0:]
            
            output = Trained_Net.forward(Target, Condition) 
            binary_output = to_binary(AP_num, output.tolist())
            output_instance.extend(binary_output)
        trained_output.append(output_instance) ###

    #print('******Loading Output data******')
    label = pd.read_csv(output_path, header=None)
    real_output = pd.DataFrame(label)
    real_output = np.array(real_output).T.tolist()
    
    count = 0
    for i in range(256):
        for j in range(test_UE_num):
            real_output_now = real_output[i][0+j*AP_num:(j+1)*AP_num]
            # real_output_now = real_output[i][0+(j+1)*AP_num:(j+2)*AP_num]
            trained_output_now = np.array(trained_output[i][0+j*AP_num:(j+1)*AP_num])
            if all(real_output_now == trained_output_now):
                pass
            else:
                count = count + 1
    acc = 1 - count/256/test_UE_num
    return acc

# test raw dataset without normalization and mapping
def test_acc_ATCNN_raw(input_path, output_path, AP_num, UE_num, test_UE_num, SNR_min):
    #print('******Loading Input data******')
    data = pd.read_csv(input_path, header=None)
    values = np.array(data).T.tolist() # with dimension of 256*300
    trained_output = []
    # test_UE_num = 1 # revise here
    
    for i in range(256):
        condition = values[i]
        condition = torch.tensor(condition)
        output_instance = []
        for j in range(test_UE_num):
            raw_condition = torch.tensor(values[i])
            condition_now = switch(raw_condition, j, AP_num+1) 
            condition_now = torch.tensor([condition_now])
            
            mirroring_condition = mapping([UE_num]*256, condition_now.tolist(), AP_num)
            
            nor_mirroring_condition = normalization(50, AP_num, mirroring_condition, 60, SNR_min, 1000) # nomalization is correct
    
            nor_mirroring_condition = torch.tensor(nor_mirroring_condition).to(torch.float32)
            
            Target = nor_mirroring_condition[..., 0:user_dim]    
            Condition = nor_mirroring_condition[..., 0:]
            
            output = Trained_Net.forward(Target, Condition) 
            binary_output = to_binary(AP_num, output.tolist())
            output_instance.extend(binary_output)
        trained_output.append(output_instance) ###

    #print('******Loading Output data******')
    label = pd.read_csv(output_path, header=None)
    real_output = pd.DataFrame(label)
    real_output = np.array(real_output).T.tolist()

    count = 0
    for i in range(256):
        for j in range(test_UE_num):
            real_output_now = real_output[i][0+j*AP_num:(j+1)*AP_num]
            trained_output_now = np.array(trained_output[i][0+j*AP_num:(j+1)*AP_num])
            if all(real_output_now == trained_output_now):
                pass
            else:
                count = count + 1
    acc = 1 - count/256/test_UE_num         
    return acc

# 4 LiFi size
user_dim = 6 # 
output_dim = 5 # 
cond_dim = 300  #
epoch = 21
SNR_min = 0

Trained_Net = ATCNN(input_dim=user_dim, cond_dim=cond_dim, cond_out_dim=user_dim, output_dim=output_dim)
device = torch.device('cpu')

Acc_list = []
for epoch_num in range(epoch):
    
    model_name = "trained_model/4LiFi_AP_SNR_Loss_Final/4LiFi_AP_epoch%s.pth"%str(epoch_num)
    
    print('******Loading Net******')
    print("Testing ATCNN model name is:", model_name)
    Trained_Net.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')), strict=False) # 1000 batch case 
    
    # confirmed that the saved net parameters are loaded and unchangable
    Trained_Net.eval()
    Trained_Net.to(device)
    print('******Starting testing different UE number******')
    
    # test dataset used in training process
    input_path = 'dataset/4LiFi_AP_SINR/input_test/nor_mirror_input_batch'
    output_path = 'dataset/4LiFi_AP_SINR/output_test/mirror_output_batch'
    
    input_path_list = []
    output_path_list = []
    
    for i in range(10):
        input_path_list.append(input_path+'%s.csv'%(str(i+1001)))
        output_path_list.append(output_path+'%s.csv'%str((i+1001)))
    
    acc_list = []
    for i in range(10):
        # UE_list = [10]*5
        UE_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        index = int(i/1)
        
        input_path = input_path_list[i]
        output_path = output_path_list[i]
        UE_num = UE_list[index]
        AP_num = 5
        acc = test_acc_ATCNN(input_path, output_path, AP_num, UE_num)
        acc_list.append(acc)
     
    print('For epoch %s, aver Acc is %s'%(epoch_num, sum(acc_list)/len(acc_list)))
    Acc_list.append(sum(acc_list)/len(acc_list))
    # save_list_to_csv(Acc_list, './figure_data/fig5_24UE_ATCNN.csv')

    
        
        
    
    
    
