# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:00:19 2023

@author: Han
"""

import numpy as np
import torch
from ATCNN_model import ATCNN_9LiFi
from utils import CsvThr, HLWNets, PA_optimization, CsvFairness
import matlab.engine
import time

print('Activating the Matlab engine')
eng = matlab.engine.start_matlab()
print('Successfully activated the Matlab engine')

###### define ATCNN parameters ######
net = HLWNets(10, 5)
csv_thr = CsvThr()
csv_fairness = CsvFairness()

if net.AP_num == 5:
    net.SNR_min = 15
    net.R_aver = 100 # Mbps
else:
    net.SNR_min = -20
    net.R_aver = 200 # Mbps
ue = 50 # max supporting UE number of ATCNN
user_dim = net.AP_num + 1 # 
output_dim = net.AP_num  # 
cond_dim = (net.AP_num+1)*(ue) #
UE_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MontoCarlotimes_list = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
######
print('******Loading Net******')
Trained_Net = ATCNN_9LiFi(input_dim=user_dim, cond_dim=cond_dim, cond_out_dim=user_dim, output_dim=output_dim)
device = torch.device('cpu')
model_name = "trained_model/Final/GT_ATCNN_9LiFi_3W_Final.pth"
Trained_Net.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')), strict=False) # 1000 batch case 
Trained_Net.eval()
Trained_Net.to(device)

for i in range(len(UE_list)):
    net.UE_num = UE_list[i] # update UE number here
    
    thr_ATCNN = []
    thr_GT = []
    thr_SSS = []
    thr_iter = []
    FL_thr = []
    
    ATCNN_fairness_list = []
    GT_fairness_list = []
    SSS_fairness_list = []
    iter_fairness_list = []
    FL_fairness_list = []
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
        ATCNN_runtime = net.load_balancing_ATCNN(Trained_Net)
        ATCNN_thr_sat = net.throughtput_calculation(net.UE_num)
        ATCNN_thr_now = ATCNN_thr_sat[0]
        ATCNN_fairness = net.Jain_fairness(ATCNN_thr_sat[1])
       
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
        GT_fairness = net.Jain_fairness(GT_thr_sat[1])
       
    ############################## SSS method #########################################
        SSS_runtime = net.load_balancing_SSS()
        SSS_thr_sat = net.throughtput_calculation(net.UE_num)
        SSS_thr_now = SSS_thr_sat[0]
        SSS_fairness = net.Jain_fairness(SSS_thr_sat[1])
       
    ################################# iterative method ###########################
        tic = time.time()
        net.load_balancing_SSS() 
        iter_results = net.load_balancing_iterative()
        toc = time.time()
        iter_runtime = toc - tic
        iter_thr_sat = net.throughtput_calculation(net.UE_num)
        iter_thr_now = iter_thr_sat[0]
        iter_fairness = net.Jain_fairness(iter_thr_sat[1])
       
    ################################## Conv FL method ###########################
        FL_runtime = net.load_balancing_FL(eng)
        FL_thr_sat = net.throughtput_calculation(net.UE_num)
        FL_thr_now = FL_thr_sat[0]
        FL_fairness = net.Jain_fairness(FL_thr_sat[1])
        
    #############################################################     
        thr_ATCNN.append(ATCNN_thr_now/1e6)
        thr_GT.append(GT_thr_now/1e6)
        thr_SSS.append(SSS_thr_now/1e6) # Mbps
        thr_iter.append(iter_thr_now/1e6)
        FL_thr.append(FL_thr_now/1e6) # Mbps
        
        ATCNN_fairness_list.append(ATCNN_fairness)
        GT_fairness_list.append(GT_fairness)
        SSS_fairness_list.append(SSS_fairness)
        iter_fairness_list.append(iter_fairness)
        FL_fairness_list.append(FL_fairness)
        
        print(j)
        
    log_thr = [net.UE_num, sum(thr_ATCNN)/len(thr_ATCNN), sum(thr_GT)/len(thr_GT), sum(thr_SSS)/len(thr_SSS), sum(thr_iter)/len(thr_iter), sum(FL_thr)/len(FL_thr)]
    log_fairness = [net.UE_num, sum(ATCNN_fairness_list)/len(ATCNN_fairness_list), sum(GT_fairness_list)/len(GT_fairness_list), sum(SSS_fairness_list)/len(SSS_fairness_list), 
                    sum(iter_fairness_list)/len(iter_fairness_list), sum(FL_fairness_list)/len(FL_fairness_list)]
    
    csv_thr.update(log_thr, 'figure_data/fig5_Thr_9LiFi_3W.csv')
    print('')
    print('For UE number %s, ATCNN thr %s, GT thr %s, SSS thr %s, iter thr %s, and FL thr %s'%(net.UE_num, log_thr[1], log_thr[2], log_thr[3], log_thr[4], log_thr[5]))
    csv_fairness.update(log_fairness, 'figure_data/fig5_Fairness_9LiFi_3W.csv')
    print('ATCNN fairness %s, GT fairness %s, SSS fairness %s, iter fairness %s, and FL fairness %s'%(log_fairness[1], log_fairness[2], log_fairness[3], log_fairness[4], log_fairness[5]))
    

