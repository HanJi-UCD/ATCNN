# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:00:19 2023

@author: Han
"""

import numpy as np
import torch
from ATCNN_model import ATCNN, to_binary, switch, ATCNN_9LiFi, global_dnn, global_dnn_9LiFi
import pandas as pd
from utils import PA_optimization, translate, mapping, normalization
import math
import csv

######
AP_size = 10 # number of WiFi+LiFi APs
ue = 50 # max supporting UE number of ATCNN
user_dim = AP_size + 1 # 
output_dim = AP_size  # 
cond_dim = (AP_size+1)*(ue) #
B = 20e6 # bps
# SNR_min = 0 # 4 LiFi, 1W case
# SNR_min = 15 # 4 LiFi, 3W case
SNR_min = -20 # 9 LiFi
mode = 'DNN'

print('******Loading Net******')
if mode == 'ATCNN':
    Trained_Net = ATCNN_9LiFi(input_dim=user_dim, cond_dim=cond_dim, cond_out_dim=user_dim, output_dim=output_dim)
    model_name = "trained_model/GT_ATCNN_9LiFi_3W_Final.pth"
else:
    Trained_Net = global_dnn_9LiFi(input_dim=cond_dim, output_dim=AP_size*ue)
    model_name = "trained_model/Final/DNN_9LiFi_50UE_CE.pth"

device = torch.device('cpu')
Trained_Net.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')), strict=False) # 1000 batch case 
# confirmed that the saved net parameters are loaded and unchangable
Trained_Net.eval()
Trained_Net.to(device)

print('******Starting testing different UE number******')
# test dataset used in training process
# input_path = 'dataset/GT_9LiFi_AP_SNR_3W/input_test_new/input_batch'
# output_path = 'dataset/GT_9LiFi_AP_SNR_3W/output_test_new/output_batch'

input_path = 'dataset/GT_9LiFi_3W_DNN/50UE/input_test_new/input_batch'
output_path = 'dataset/GT_9LiFi_3W_DNN/50UE/output_test_new/output_batch'

input_path_list = []
output_path_list = []
for i in range(5):
    input_path_list.append(input_path+'%s.csv'%(str(i+1001)))
    output_path_list.append(output_path+'%s.csv'%str((i+1001)))

gap_list = []
thr_dataset_Final = []
thr_ATCNN_Final = []
for i in range(2):
    # UE_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    UE_list = [50]*5
    
    input_path = input_path_list[i]
    output_path = output_path_list[i]
    UE_num = UE_list[i]
    
    #print('******Loading Input data******')
    input_data = pd.read_csv(input_path, header=None)
    input_data = np.array(input_data).T.tolist() # with dimension of 256*300
    
    #print('******Loading Output data******')
    output_data = pd.read_csv(output_path, header=None)
    output_data = np.array(output_data).T.tolist()
    
    thr_dataset_list = []
    thr_ATCNN_list = []
    
    for j in range(256):
        data1 = input_data[j]
        data2 = output_data[j]
        
        ############ for dataset throughput ############
        nested_data2 = [data2[ii:ii+AP_size] for ii in range(0, len(data2), AP_size)]
        X_iu_dataset = translate(nested_data2)
        R_requirement = []
        Capacity = []
        for jj in range(UE_num):
            R_requirement.append(data1[(jj+1)*(AP_size+1)-1])
            SNR_now = data1[0+jj*(AP_size+1):AP_size+jj*(AP_size+1)]
            Capacity_now = [B*math.log2(1+10**(snr/10)) for snr in SNR_now]
            Capacity.append(Capacity_now)
            
        Rho_iu = PA_optimization(AP_size, UE_num, X_iu_dataset, R_requirement, Capacity, 1)
        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num   
        
        thr_dataset = []
        for jj in range(UE_num):
            list1 = Capacity[jj]
            list2 = Rho_transposed[jj]
            thr_dataset.append(min(sum(list(np.multiply(list1, list2))), R_requirement[jj]))
        thr_dataset_list.append(sum(thr_dataset))
        
        ############ for ATCNN throughput ############
        binary_output = []
        if mode == 'ATCNN':
            for jj in range(UE_num):
                raw_condition = torch.tensor(data1)
                condition_now = switch(raw_condition, jj, AP_size+1) 
                condition_now = torch.tensor([condition_now])
                
                mirroring_condition = mapping([UE_num]*256, condition_now.tolist(), AP_size)
                
                nor_mirroring_condition = normalization(50, AP_size, mirroring_condition, 60, SNR_min, 1000) # nomalization is correct
        
                nor_mirroring_condition = torch.tensor(nor_mirroring_condition).to(torch.float32)
                
                Target = nor_mirroring_condition[..., 0:user_dim]    
                Condition = nor_mirroring_condition[..., 0:]
                
                output = Trained_Net.forward(Target, Condition) 
                binary_output_now = to_binary(AP_size, output.tolist())
                binary_output.append(binary_output_now)
            
        else: # for DNN mode
            data1 = input_data[j]
            nor_input = normalization(UE_num, AP_size, [data1], 60, SNR_min, 1000)
            input_data_now = torch.tensor(nor_input).to(torch.float32)
            output_data_now = Trained_Net(input_data_now) 
            
            binary_output = []
            for jj in range(UE_num):       
                output_now = output_data_now[0][0+jj*AP_size:(jj+1)*AP_size].tolist()
                binary_output_now = to_binary(AP_size, [output_now])
                binary_output.append(binary_output_now)

        X_iu_NN = translate(binary_output)
            
        Rho_iu = PA_optimization(AP_size, UE_num, X_iu_NN, R_requirement, Capacity, 1)
        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num   
        
        thr_ATCNN = []
        for jj in range(UE_num):
            list1 = Capacity[jj]
            list2 = Rho_transposed[jj]
            thr_ATCNN.append(min(sum(list(np.multiply(list1, list2))), R_requirement[jj]))
        thr_ATCNN_list.append(sum(thr_ATCNN))
    
    thr_dataset_Final.append(sum(thr_dataset_list)/len(thr_dataset_list))
    thr_ATCNN_Final.append(sum(thr_ATCNN_list)/len(thr_ATCNN_list))
    gap_now = 1 - (sum(thr_ATCNN_list)/len(thr_ATCNN_list))/(sum(thr_dataset_list)/len(thr_dataset_list))
    gap_list.append(gap_now)
    print('For UE number %s, Gap is %s'%(UE_num, gap_now))

# Combine the lists into a list of rows
# rows = zip(thr_dataset_Final, thr_ATCNN_Final, gap_list)
# # Write the data to the CSV file
# with open('./figure_data/fig4_9LiFi_Gap_3W.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Dataset throughput(bps)', 'ATCNN throughput(bps)', 'Gap'])  # Write header
#     writer.writerows(rows)  # Write rows