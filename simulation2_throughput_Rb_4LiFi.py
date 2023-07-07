# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:00:19 2023

@author: Han
"""

import numpy as np
import torch
from ATCNN_model import ATCNN, global_dnn
from utils import CsvThr, HLWNets, PA_optimization
import matlab.engine
import time

print('Activating the Matlab engine')
eng = matlab.engine.start_matlab()
print('Successfully activated the Matlab engine')

###### define ATCNN parameters ######
net = HLWNets(5, 5)
csv = CsvThr()

if net.AP_num == 5:
    net.SNR_min = 15
else:
    net.SNR_min = -20
ue = 50 # max supporting UE number of ATCNN
UE = 50 # UE number

user_dim = net.AP_num + 1 # 
output_dim = net.AP_num  # 
cond_dim = (net.AP_num+1)*(ue) #

R = [10, 20, 50, 100, 150, 200]
MontoCarlotimes_list = [50, 50, 50, 50, 50, 50]
######
print('******Loading ATCNN Net******')
Trained_Net_ATCNN = ATCNN(input_dim=user_dim, cond_dim=cond_dim, cond_out_dim=user_dim, output_dim=output_dim)
# Trained_Net = ATCNN_9LiFi(input_dim=user_dim, cond_dim=cond_dim, cond_out_dim=user_dim, output_dim=output_dim)
device = torch.device('cpu')
model_name = "trained_model/Final/GT_ATCNN_4LiFi_3W_Final.pth"
Trained_Net_ATCNN.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')), strict=False) # 1000 batch case 
Trained_Net_ATCNN.eval()
Trained_Net_ATCNN.to(device)


DNN_input_dim = (net.AP_num+1)*UE
DNN_output_dim = net.AP_num*UE # 
print('******Loading DNN Net******')
Trained_Net_DNN = global_dnn(input_dim=DNN_input_dim, output_dim=DNN_output_dim)
device = torch.device('cpu')
model_name = "trained_model/Final/DNN_4LiFi_%sUE_CE.pth"%(str(UE))
Trained_Net_DNN.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')), strict=False) # 1000 batch case 
Trained_Net_DNN.eval()
Trained_Net_DNN.to(device)

for i in range(len(R)):
    net.UE_num = UE # update UE number here
    net.R_aver = R[i]
    
    thr_ATCNN = []
    thr_GT = []
    thr_SSS = []
    thr_iter = []
    thr_FL = []
    thr_DNN = []
    
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
    ################################# ATCNN method ##############################
        ATCNN_runtime = net.load_balancing_ATCNN(Trained_Net_ATCNN)
        ATCNN_thr_sat = net.throughtput_calculation(net.UE_num)
        ATCNN_thr_now = ATCNN_thr_sat[0]
            
    ################################# GT method ##################################
        tic = time.time()
        net.load_balancing_SSS() # use SSS as initial X_iu
        # calculate initial satisfaction list
        Rho_iu = PA_optimization(net.AP_num, net.UE_num, net.X_iu, net.R_requirement, net.Capacity, 1)
        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
        sat_list = []
        for ii in range(net.UE_num):
            list1 = net.Capacity[ii]
            list2 = Rho_transposed[ii]
            sat_now = min(sum(list(np.multiply(list1, list2)))/net.R_requirement[ii], 1)      
            sat_list.append(sat_now)
        GT_results = net.load_balancing_GT(sat_list)
        toc = time.time()
        GT_runtime = toc - tic
        GT_thr_sat = net.throughtput_calculation(net.UE_num)
        GT_thr_now = GT_thr_sat[0]
            
    ############################## SSS method #########################################
        SSS_runtime = net.load_balancing_SSS()
        SSS_thr_sat = net.throughtput_calculation(net.UE_num)
        SSS_thr_now = SSS_thr_sat[0]
            
    ################################# iterative method ###########################
        tic = time.time()
        net.load_balancing_SSS() 
        iter_results = net.load_balancing_iterative()
        toc = time.time()
        iter_runtime = toc - tic
        iter_thr_sat = net.throughtput_calculation(net.UE_num)
        iter_thr_now = iter_thr_sat[0]
            
    ################################## Conv FL method ###########################
        FL_runtime = net.load_balancing_FL(eng)
        FL_thr_sat = net.throughtput_calculation(net.UE_num)
        FL_thr_now = FL_thr_sat[0]
            
    ################################# DNN method ##############################
        net.load_balancing_DNN(Trained_Net_DNN)
        DNN_results = net.throughtput_calculation(net.UE_num)
        
    #############################################################    
        thr_ATCNN.append(ATCNN_thr_now/1e6)
        thr_GT.append(GT_thr_now/1e6)
        thr_SSS.append(SSS_thr_now/1e6) # Mbps
        thr_iter.append(iter_thr_now/1e6)
        thr_FL.append(FL_thr_now/1e6)
        thr_DNN.append(DNN_results[0]/1e6)
        
        print(j)
        
    log = [net.R_aver, sum(thr_ATCNN)/len(thr_ATCNN), sum(thr_GT)/len(thr_GT), sum(thr_SSS)/len(thr_SSS), sum(thr_iter)/len(thr_iter), sum(thr_FL)/len(thr_FL), sum(thr_DNN)/len(thr_DNN)]

    csv.update(log, 'figure_data/fig7_ThrRb_4LiFi_3W_50UE.csv')
    print('For Aver R %s, ATCNN thr %s, GT thr %s, SSS thr %s, iter thr %s, FL thr %s, and DNN thr %s'%(net.R_aver, log[1], log[2], log[3], log[4], log[5], log[6]))
    

