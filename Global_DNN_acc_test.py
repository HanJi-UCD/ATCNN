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
from utils import mapping, normalization
from ATCNN_model import global_dnn, to_binary
warnings.filterwarnings("ignore")

# test normalised and mirrored dataset
def test_acc_DNN(input_path, output_path, AP_num, UE_num):
    #print('******Loading Input data******')
    data = pd.read_csv(input_path, header=None)
    values = np.array(data).T.tolist() # with dimension of 256*300
    trained_output = []
    test_UE_num = UE_num # revise here
    
    input_data = torch.tensor(values).to(torch.float32) # 
    output_data = Trained_Net(input_data) 
    
    for i in range(256):
        output = output_data[i, ...]
        output_instance = []
        for j in range(test_UE_num):       
            output_now = output[0+j*AP_num:(j+1)*AP_num].tolist()
            binary_output = to_binary(AP_num, [output_now])
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
AP_size = 5 # number of WiFi+LiFi APs
ue = 50 # max supporting UE number
input_dim = (AP_size+1)*ue
output_dim = AP_size*ue # 

Trained_Net = global_dnn(input_dim=input_dim, output_dim=output_dim)
device = torch.device('cpu')

model_name = "trained_model/Final/DNN_4LiFi_50UE_CE.pth"

print('******Loading Net******')
print("Testing ATCNN model name is:", model_name)
Trained_Net.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')), strict=False) # 1000 batch case 

# confirmed that the saved net parameters are loaded and unchangable
Trained_Net.eval()
Trained_Net.to(device)
print('******Starting testing different UE number******')

# test dataset used in training process
input_path = 'dataset/GT_4LiFi_3W_DNN/50UE/input_test/nor_input_batch'
output_path = 'dataset/GT_4LiFi_3W_DNN/50UE/output_test/output_batch'

input_path_list = []
output_path_list = []

for i in range(5):
    input_path_list.append(input_path+'%s.csv'%(str(i+1001)))
    output_path_list.append(output_path+'%s.csv'%str((i+1001)))

acc_list = []
for i in range(5):
    UE_list = [50]*5
    
    input_path = input_path_list[i]
    output_path = output_path_list[i]
    UE_num = UE_list[i]
    
    AP_num = 5
    acc = test_acc_DNN(input_path, output_path, AP_num, UE_num)
    acc_list.append(acc)
    print('UE number is:', UE_num, 'and Accuracy is:', acc)
 
print('Aver Acc is:', sum(acc_list)/len(acc_list))

    
        
        
    
    
    
