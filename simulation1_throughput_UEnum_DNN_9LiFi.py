# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:00:19 2023

@author: Han
"""

import numpy as np
import torch
from ATCNN_model import global_dnn_9LiFi as global_dnn
from utils import CsvDNN, HLWNets, PA_optimization, CsvFairness
import matlab.engine
import time

print('Activating the Matlab engine')
eng = matlab.engine.start_matlab()
print('Successfully activated the Matlab engine')

###### define ATCNN parameters ######
net = HLWNets(10, 5)
csv_thr = CsvDNN()
csv_fairness = CsvFairness()

if net.AP_num == 5:
    net.SNR_min = 15 # for 3W LiFi case
    net.R_aver = 100 # Mbps
    
else:
    net.SNR_min = -20
    net.R_aver = 200 # Mbps

AP_size = net.AP_num # number of WiFi+LiFi APs

UE_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MontoCarlotimes_list = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

for i in range(len(UE_list)):
    net.UE_num = UE_list[i] # update UE number here
    ue = net.UE_num
    input_dim = (AP_size+1)*ue
    output_dim = AP_size*ue # 
    
    print('******Loading Net******')
    Trained_Net = global_dnn(input_dim=input_dim, output_dim=output_dim)

    device = torch.device('cpu')
    model_name = "trained_model/Final/DNN_9LiFi_%sUE_CE.pth"%(str(ue))
    Trained_Net.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')), strict=False) # 1000 batch case 
    Trained_Net.eval()
    Trained_Net.to(device)
    
    thr_DNN = []
    DNN_fairness_list = []
    
    for j in range(MontoCarlotimes_list[i]):
        ###### UE distribution and SNR calculation ######
        net.UE_position = []
        for k in range(net.UE_num):    
            net.UE_position.append([np.random.rand(1).tolist()[0]*net.X_length, np.random.rand(1).tolist()[0]*net.Y_length, 0])
        R_requirement = np.random.gamma(1, net.R_aver, net.UE_num)*1e6 # bps
        R_requirement = np.clip(R_requirement, 1*1e6, 1000*1e6)
        
        net.R_requirement = R_requirement.tolist()
        
        ###### calculate SNR and capacity
        net.snr_calculation()
    ################################# DNN method ##############################
        net.load_balancing_DNN(Trained_Net)
        DNN_results = net.throughtput_calculation(net.UE_num)
        sat_list = DNN_results[1]
        fairness = net.Jain_fairness(sat_list)
        
        thr_DNN.append(DNN_results[0])
        DNN_fairness_list.append(fairness)
        
        print(j)
        
    log = [net.UE_num, sum(thr_DNN)/len(thr_DNN), sum(DNN_fairness_list)/len(DNN_fairness_list)]
    
    csv_thr.update(log, 'figure_data/DNN_Thr_Fairness_9LiFi_3W_CE1.csv')
    print('')
    print('For UE number %s, DNN thr %s, and fairness is %s'%(net.UE_num, log[1], log[2]))
    

