#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:23:52 2022

@author: Han
"""
import numpy as np
from copy import copy,deepcopy
import pandas as pd
import torch
import time
import argparse
import math
import csv
from scipy.optimize import minimize
from torch.utils.data import Dataset
import matlab.engine
import random

class HLWNets():
    def __init__(self, AP_num, UE_num):     
        self.AP_num = AP_num
        self.UE_num = UE_num
        self.B = 20*1e6 # unit: bps
        if AP_num == 5:
            self.X_length = 5 # room size
            self.Y_length = 5 # room size
            self.Z_height = 3 # room height
            self.AP_position = [[2.5, 2.5, 0.5], [1.25, 1.25, 3], [3.75, 1.25, 3], [1.25, 3.75, 3], [3.75, 3.75, 3]]
        else:
            self.X_length = 9 # room size
            self.Y_length = 9 # room size
            self.Z_height = 3 # room height
            self.AP_position = [[4.5, 4.5, 0.5], [1.5, 1.5, 3], [4.5, 1.5, 3], [7.5, 1.5, 3], 
                   [1.5, 4.5, 3], [4.5, 4.5, 3], [7.5, 4.5, 3],
                   [1.5, 7.5, 3], [4.5, 7.5, 3], [7.5, 7.5, 3]]
    
    def snr_calculation(self):
        self.SNR_matrix = []
        self.Capacity = []
        for k in range(self.UE_num):
            snr_list = []
            for kk in range(self.AP_num):
                if kk == 0:
                    mode = 'WiFi'
                else:
                    mode = 'LiFi'
                snr = SNR_calculation(self.AP_position[kk], self.UE_position[k], mode, self.AP_num-1) #
                snr = max(snr, 0.01) # set snr domain meaningful, > 0
                snr_dB = 10*math.log10(snr) # in dB scale
                snr_dB = max(snr_dB, -20) # set -20dB as min value
                snr_list.append(snr_dB)
            ################### This is my error correction ###################
            ################### Please note this capacity calculation is based on the basic Shannon Capacity Formula ###################
            ################### For a more accurate capacity, please use another LiFi capacity formula given in paper ###################
            Capacity_vector = [self.B*math.log2(1 + 10**(j/10)) for j in snr_list]
            ###################################################################
            self.SNR_matrix.append(snr_list)
            self.Capacity.append(Capacity_vector)
    
    def throughtput_calculation(self, UE_num):
        Rho_iu = PA_optimization(self.AP_num, self.UE_num, self.X_iu, self.R_requirement, self.Capacity, 1)
        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num   
        thr_dataset = []
        satisfaction_list = []
        for jj in range(UE_num):
            list1 = self.Capacity[jj]
            list2 = Rho_transposed[jj]
            thr_dataset.append(min(sum(list(np.multiply(list1, list2))), self.R_requirement[jj]))
            satisfaction_list.append(min(sum(list(np.multiply(list1, list2)))/self.R_requirement[jj], 1))
        sum_thr = sum(thr_dataset)
        results = [sum_thr, satisfaction_list]
        return results
    
    def Jain_fairness(self, sat_list):
        list_now = np.array(sat_list)**2
        fairness = (sum(sat_list))**2/(self.UE_num*sum(list_now.tolist()))
        return fairness
    
    def load_balancing_DNN(self, Trained_Net):
        # load DNN model
        values = []
        for i in range(self.UE_num):
            value_now = self.SNR_matrix[i].copy()
            value_now.append(self.R_requirement[i])
            values.append(value_now)
        Combined_input = sum(values, []) # flatten the list
        nor_input = normalization(self.UE_num, self.AP_num, [Combined_input], 60, self.SNR_min, 1000)
        nor_input = torch.tensor(nor_input).to(torch.float32)
        output = Trained_Net(nor_input) 
        binary_output = []
        for jj in range(self.UE_num):       
            output_now = output[0][0+jj*self.AP_num:(jj+1)*self.AP_num].tolist()
            binary_output_now = to_binary(self.AP_num, [output_now])
            binary_output.append(binary_output_now)
        self.X_iu = translate(binary_output)
    
    def load_balancing_ATCNN(self, Trained_Net):
        # load ATCNN model
        X_iu = []
        values = []
        for i in range(self.UE_num):
            value_now = self.SNR_matrix[i].copy()
            value_now.append(self.R_requirement[i])
            values.append(value_now)
        Combined_input = sum(values, []) # flatten the list
        
        tic = time.time()
        for j in range(self.UE_num):
            raw_condition = torch.tensor(Combined_input)
            
            condition_now = switch(raw_condition, j, self.AP_num+1) # switch j-th UE into last position
            
            mirroring_condition = mapping([self.UE_num], [condition_now.tolist()], self.AP_num)
            
            nor_mirroring_condition = normalization(50, self.AP_num, mirroring_condition, 60, self.SNR_min, 1000) # nomalization is correct
    
            nor_mirroring_condition = torch.tensor(nor_mirroring_condition).to(torch.float32)
            
            Target = nor_mirroring_condition[..., 0:self.AP_num+1]    
            
            output = Trained_Net.forward(Target, nor_mirroring_condition)
            
            binary_output = to_binary(self.AP_num, output.tolist())
            
            X_iu.append(binary_output)
        
        self.X_iu = translate(X_iu)
        toc = time.time()
        return (toc-tic)

    def load_balancing_SSS(self):
        # do SSS LB only for target UE
        # get target
        tic = time.time()
        X_iu = []
        for i in range(self.UE_num):
            SNR_list = self.SNR_matrix[i]
            max_SNR = max(SNR_list)
            index = SNR_list.index(max_SNR)
            X_iu.append(index + 1)
        toc = time.time()
        self.X_iu = X_iu
        return (toc-tic)

    def load_balancing_GT(self, Satisfaction_vector):
        N_f = 5  # number of AP candidates for each UE
        count = 0
        mode = 0
        payoff_vector = [0]
        N = self.UE_num
        X_iu_old = self.X_iu
        
        tic = time.time()
        
        while mode <= N:
            estimated_payoff = [0]*N_f
            mutation_probability = np.zeros(self.UE_num)
            aver_payoff = sum(Satisfaction_vector)/self.UE_num
            
            for i in range(self.UE_num):
                if Satisfaction_vector[i] < aver_payoff:
                    mutation_probability[i] = 1 - Satisfaction_vector[i]/aver_payoff
                else:
                    mutation_probability[i] = 0
                
                x = np.random.rand(1)
                # apply mutation rule here
                if x < mutation_probability[i]:
                    old_AP = X_iu_old[i]
                    
                    # find 5 AP candidates for mutation UE
                    SNR_list = self.SNR_matrix[i]
                    AP_index = sorted(range(len(SNR_list)), key=lambda i: SNR_list[i], reverse=True)[:5]
                    
                    for j in range(N_f):
                        X_iu_old[i] = AP_index[j] + 1 # update X_iu
                        Rho_iu = PA_optimization(self.AP_num, self.UE_num, X_iu_old, self.R_requirement, self.Capacity, 0)
                        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
                        
                        sat_list = []
                        for ii in range(self.UE_num):
                            list1 = self.Capacity[ii]
                            list2 = Rho_transposed[ii]
                            sat_now = min(sum(list(np.multiply(list1, list2)))/self.R_requirement[ii], 1)      
                            sat_list.append(sat_now)

                        estimated_payoff[j] = sum(sat_list)/self.UE_num
                        
                    AP_muted = [AP_index[ii] for ii in range(N_f) if estimated_payoff[ii] == max(estimated_payoff)]
                    
                    if len(AP_muted) > 1:
                        SNR_set = [self.SNR_matrix[i][k] for k in AP_muted]
                        index = SNR_set.index(max(SNR_set))
                        AP_muted_chosen = AP_muted[index]
                        X_iu_old[i] = AP_muted_chosen + 1 # update X_iu here
                    else:
                        X_iu_old[i] = AP_muted[0] + 1 # update X_iu here
                    
                    AP_mutated = X_iu_old[i]
                    
                    if AP_mutated != old_AP:
                        Rho_iu = PA_optimization(self.AP_num, self.UE_num, X_iu_old, self.R_requirement, self.Capacity, 0)
                        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
                        
                        Satisfaction_vector = []
                        for ii in range(self.UE_num):
                            list1 = self.Capacity[ii]
                            list2 = Rho_transposed[ii]
                            sat_now = min(sum(list(np.multiply(list1, list2)))/self.R_requirement[ii], 1)      
                            Satisfaction_vector.append(sat_now)

                        aver_payoff = sum(Satisfaction_vector)/self.UE_num
                        
                        if aver_payoff > payoff_vector[-1]:
                            mode = 0
                            payoff_vector.append(aver_payoff)
                        else:
                            mode += 1
                    else:
                        mode += 1
                else:
                    # no mutation action for UE j
                    payoff_vector.append(payoff_vector[-1])
                    
                if mode > N:
                    break
                count += 1
            if aver_payoff >= 0.99:
                break
        toc = time.time()
        
        self.X_iu = X_iu_old
        results = [aver_payoff, count, payoff_vector, toc-tic]
        return results

    def load_balancing_iterative(self):
        # iterative method
        # SSS as initial AP connections
        tic = time.time()
        X_iu = self.X_iu
        N = self.UE_num
        count = 0
        num = 0 # total iteration times
        N_f = 5 # AP candidates
        while count <= N:
            order = random.sample(range(0, self.UE_num), self.UE_num)
            for ii in range(self.UE_num):
                i = order[ii]
                SNR_list = self.SNR_matrix[i]
                AP_index = sorted(range(len(SNR_list)), key=lambda i: SNR_list[i], reverse=True)[:N_f]
                object_function_list = []
                index_old = X_iu[i]
                for j in range(N_f):
                    X_iu[i] = AP_index[j] + 1 # update X_iu
                    Rho_iu = PA_optimization(self.AP_num, self.UE_num, X_iu, self.R_requirement, self.Capacity, 0)
                    Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
                    sat_list = []
                    for ii in range(self.UE_num):
                        list1 = self.Capacity[ii]
                        list2 = Rho_transposed[ii]
                        sat_now = min(sum(list(np.multiply(list1, list2)))/self.R_requirement[ii], 1)      
                        sat_list.append(sat_now)
                    object_function_list.append(np.sum(np.log(sat_list)))
                chosen_AP = object_function_list.index(max(object_function_list)) 
                X_iu[i] = AP_index[chosen_AP] + 1           
                if X_iu[i] == index_old:
                    count = count + 1
                else:
                    count = 0
                num = num + 1
        toc = time.time()
        self.X_iu = X_iu
        toc = time.time()
        results = [num, toc-tic]
        return results

    def load_balancing_FL(self, eng):
        # two-stage FL method for LB
        # Stage1: Choose UE connects to LiFi or WiFi
        # Stage2: Perform LB using SSS in stand-alone LiFi or WiFi network
        tic = time.time()
        Rb = self.R_aver
        # FL_rule_threshold = [[0, 0, Rb, 2*Rb, 10000], [15, 40, 50, 60, 70], [20, 22, 25, 27, 30], [0, 0.2, 0.5, 0.8, 1], [0, 0.2, 0.5, 0.8, 1]] # 3W case
        FL_rule_threshold = [[0, 0, Rb, 2*Rb, 10000], [20, 40, 50, 60, 70], [30, 32, 35, 37, 38.5], [0, 0.2, 0.5, 0.8, 1], [0, 0.2, 0.5, 0.8, 1]] # 3W case
        FL_rule_threshold = matlab.double(FL_rule_threshold)
        SNR_matrix = np.array(self.SNR_matrix).T.tolist()
        SNR_matrix = matlab.double(SNR_matrix)
        R_requirement = matlab.double(self.R_requirement)
        X_iu = eng.Conv_FL(self.UE_num, self.B, SNR_matrix, R_requirement, FL_rule_threshold)
        X_iu = np.array(X_iu, dtype=int).tolist()
        self.X_iu = X_iu[0]
        toc = time.time()
        return (toc-tic)

def cluster_distribution(net, cluster_num, N):
    # N: hot-users conneted with each cluster
    length = net.X_length
    if cluster_num == 1:
        hot_user = multivariate_random(length, N)
    elif cluster_num == 2:
        hot_user1 = multivariate_random(length, N)
        hot_user2 = multivariate_random(length, N)
        hot_user = np.concatenate((hot_user1, hot_user2), axis=0)
    elif cluster_num == 3:    
        hot_user1 = multivariate_random(length, N)
        hot_user2 = multivariate_random(length, N)
        hot_user3 = multivariate_random(length, N)
        hot_user = np.concatenate((hot_user1, hot_user2, hot_user3), axis=0)
    elif cluster_num == 4:
        hot_user1 = multivariate_random(length, N)
        hot_user2 = multivariate_random(length, N)
        hot_user3 = multivariate_random(length, N)
        hot_user4 = multivariate_random(length, N)
        hot_user = np.concatenate((hot_user1, hot_user2, hot_user3, hot_user4), axis=0)
    else:
        hot_user1 = multivariate_random(length, N)
        hot_user2 = multivariate_random(length, N)
        hot_user3 = multivariate_random(length, N)
        hot_user4 = multivariate_random(length, N)
        hot_user5 = multivariate_random(length, N)
        hot_user = np.concatenate((hot_user1, hot_user2, hot_user3, hot_user4, hot_user5), axis=0)
    hot_user = hot_user.tolist()
    for k in range(net.UE_num - cluster_num*N):    
        hot_user.append([np.random.rand(1).tolist()[0]*net.X_length, np.random.rand(1).tolist()[0]*net.Y_length, 0])
    return hot_user

def multivariate_random(length, N):
    # hot spot area is 0.5 meter away from the bounary
    x = (length - 1) * np.random.rand(1, 2) + 0.5
    X = np.maximum(np.minimum(np.random.normal(x[0, 0], 0.25, N), length - 0.1), 0.1)
    Y = np.maximum(np.minimum(np.random.normal(x[0, 1], 0.25, N), length - 0.1), 0.1)
    Z = np.zeros(N)
    hot_user = np.column_stack((X, Y, Z))
    return hot_user



def loader(input_path,
           output_path,
           batch_size,
           data_type='default',
           req_norm = False,
           shuffle = True,
           ):
    if not req_norm:
        # transform = transforms.ToTensor()
        input_data = torch.load(input_path)
        output_data = torch.load(output_path)
        dataset = data_handler(input_data, output_data,transform=None)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=shuffle)
        return loader
    
class data_handler(Dataset):
    def __init__(self,input_data,output_data,transform=None): 
        self.input_data = input_data
        self.output_data = output_data
        self.transform = transform 
        assert self.input_data.shape[0] == self.output_data.shape[0], 'input and output cannot match'
    
    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self,index):
        data_in = self.input_data[index]
        data_out = self.output_data[index]
        if self.transform:
            data_in = self.transform(data_in)
            data_out = self.transform(data_out)
        data = [data_in, data_out]
        return data    

class Csvloss():
    def __init__(self,
                 ):
        self.title = ['epoch','step','loss_train','loss_test']
        self.data = []
        
    def update(self, log, path):
        self.data.append(log)
        dataframe = pd.DataFrame(data=self.data,columns=self.title)
        dataframe.to_csv(path,index=False,sep=',')
        
class CsvThr():
    def __init__(self,
                 ):
        self.title = ['UE_num/Rb','ATCNN Throughput (Mbps)','GT Throughput (Mbps)','SSS Throughput (Mbps)', 'iterative Throughput (Mbps)', 'FL Throughput (Mbps)', 'DNN Throughput (Mbps)']
        self.data = []
        
    def update(self, log, path):
        self.data.append(log)
        dataframe = pd.DataFrame(data=self.data, columns=self.title)
        dataframe.to_csv(path,index=False, sep=',')

class CsvDNN():
    def __init__(self,
                 ):
        self.title = ['UE number','Thr','Fairness']
        self.data = []
        
    def update(self, log, path):
        self.data.append(log)
        dataframe = pd.DataFrame(data=self.data,columns=self.title)
        dataframe.to_csv(path,index=False,sep=',')       
        
class CsvDNN_New():
    def __init__(self,
                 ):
        self.title = ['UE number','Rb','Thr','Fairness']
        self.data = []
        
    def update(self, log, path):
        self.data.append(log)
        dataframe = pd.DataFrame(data=self.data,columns=self.title)
        dataframe.to_csv(path,index=False,sep=',')   


class CsvFairness():
    def __init__(self,
                 ):
        self.title = ['UE_num/Rb','ATCNN Jains Fairness','GT Jains Fairness','SSS Jains Fairness', 'iterative Jains Fairness', 'FL Jains Fairness']
        self.data = []
        
    def update(self, log, path):
        self.data.append(log)
        dataframe = pd.DataFrame(data=self.data, columns=self.title)
        dataframe.to_csv(path,index=False, sep=',')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='batch size')
    parser.add_argument('--log-freq', type=int, default=20, help='weight-decay')
    parser.add_argument('--eval-freq', type=int, default=50, help='weight-decay')
    args = parser.parse_args()
    return args

def mapping(UEnum_list, dataset, AP_size):
    mapping_condition = []
    for i in range(len(dataset)):
        sub_dataset = dataset[i][0:UEnum_list[i]*(AP_size+1)]
        target = sub_dataset[0:AP_size+1] # keep target
        condition = sub_dataset[AP_size+1:] # do mirroring operation for condition except for target UE
        mirroring_condition = mirror(49, AP_size, UEnum_list[i]-1, condition)
        new_condition = sum([target, mirroring_condition], []) # combine target and condition list
        mapping_condition.append(new_condition)
    return mapping_condition

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
    
def normalization(M, AP_num, dataset, SNR_max, SNR_min, R_max):
    # input data is 2D list with dimension: 256*300
    nor_data_list = []
    for j in range(len(dataset)):
        condition = dataset[j]
        data = np.reshape(condition, (M, AP_num+1)) # array
        nor_data = []
        for i in range(M):
            target = data[i]
            SNR = list((target[0:AP_num]-SNR_min)/(SNR_max - SNR_min)) # 0-1
            R = target[AP_num]/1e6 # Mbps
            R = (math.log10(R)/(math.log10(R_max))) # 0-1
            SNR.append(R)
            nor_data.append(SNR)
        nor_data = sum(nor_data, [])    
        nor_data_list.append(nor_data)
    return nor_data_list

def switch(ipt,idx,user_dim):   #swtich the user data on idx to the first position
    temp = deepcopy(ipt[...,idx*user_dim:idx*user_dim+user_dim])
    ipt[...,idx*user_dim:idx*user_dim+user_dim] = deepcopy(ipt[...,0:user_dim])
    ipt[...,0:user_dim] = temp
    return ipt

def save_list_to_csv(data_list, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_list)
        
def PA_optimization(AP_num, UE_num, X_iu_list, R_required, Capacity, opt_mode):
    X_iu = np.zeros((AP_num, UE_num), dtype=int)
    for i in range(UE_num):
        X_iu[X_iu_list[i]-1, i] = 1

    Rho_iu = np.zeros((AP_num, UE_num))
    for i in range(AP_num):
        connected_UE = np.where(X_iu[i,:] == 1)[0].tolist()
        if len(connected_UE) != 0:
            if opt_mode == 1:
                A = np.ones((1, len(connected_UE)))
                b = 1
                lb = np.zeros((len(connected_UE),))
                ub = np.ones((len(connected_UE),))
                X0 = (lb+ub)/(len(connected_UE)+1)
                options = {"maxiter": 10000, "ftol": 1e-10}
                Capacity_list = [Capacity[j][i] for j in connected_UE]
                R_list = [R_required[j] for j in connected_UE]
                res = minimize(object_function, X0, args=(Capacity_list, R_list), method="SLSQP", bounds=list(zip(lb,ub)), constraints={"type": "eq", "fun": lambda x: np.dot(A, x) - b}, options=options)
                X = res.x
                Rho_iu[i, connected_UE] = X
            else:
                Rho_iu[i, connected_UE] = np.ones((len(connected_UE))) / len(connected_UE)
    

    return Rho_iu

def object_function(X, Capacity, R_required):
    cost = 0
    for i in range(len(Capacity)):
        cost += np.log2(min(Capacity[i]*X[i]/R_required[i], 1))
    cost = -cost
    return cost

# output X_iu for each UE
def translate(binray_output):
    X_iu = []
    for i in range(len(binray_output)): 
        AP_index = binray_output[i].index(1) + 1
        X_iu.append(AP_index)
    return X_iu

def to_binary(AP_num, vector):
    vector = vector[0]
    binary_vector = []
    for i in range(AP_num):
        if vector[i] == max(vector):
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    return binary_vector

def distance(A, B):
    dis = math.sqrt(( float(A[0]) - float(B[0]))**2 + (float(A[1]) - float(B[1]))**2 + (float(A[2]) - float(B[2]))**2)    
    return dis

def SNR_calculation(AP_position, UE_position, mode, AP_size):
    if mode == 'LiFi':
        # LiFi parameters
        if AP_size == 4:
            X_length = 5
            Y_length = 5
        elif AP_size == 9:
            X_length = 9
            Y_length = 9
        else:
            X_length = 16
            Y_length = 16
        Z_height = 3
        P_mod = 3 # Modulated power
        N_0 = 10**(-21) # Noise power spectral density: Watt/Hz
        Phi = 1.0472 # semiangle: radian
        # Phi = (math.pi)/3 # semiangle: radian
        FOV = 1.3963 # FOV: radian
        # FOV = 80*(math.pi)/180 # FOV: radian
        n = 1.5 # Reflective index of concentrator
        R_pd = 0.53 # PD responsivity
        # k = 0.5; #  optical to electric conversion coefficient
        A = 0.0001 # Detector area: m**2
        m = 1 # Lambertian order
        B = 20000000
        Ka = 0.8
        # LOS
        d_LOS = distance(AP_position, UE_position)
        cos_phi = (Z_height - UE_position[2])/d_LOS
        if abs(math.acos(cos_phi)) <= Phi:
           H_LOS = (m+1)*A*n**2*Z_height**(m+1) / (2*math.pi*(math.sin(FOV))**2*(d_LOS**(m+3))) # correct          
        else:
           H_LOS = 0
        # NLOS
        H_NLOS = Capacity_NLOS(AP_position[0], AP_position[1], Z_height, UE_position[0], UE_position[1], X_length, Y_length, Z_height) # call sub-function
        SNR = (R_pd*P_mod*Ka*(H_LOS + H_NLOS))**2/N_0/B
        # print('LOS is %s, NLOS is %s,SNR is %s'%(H_LOS, H_NLOS, SNR))
    else:
        # WiFi
        d_LOS = distance(AP_position, UE_position)
        radiation_angle = math.acos(0.5/d_LOS) # radian unit
        P_WiFi_dBm = 20
        P_WiFi = 10**(P_WiFi_dBm/10)/1000 # 20 dBm, convert to watts: 0.1 W
        B_WiFi = 20000000 # 20 MHz
        N_WiFi = -174 # dBm/Hz
        N_WiFi = 10**(N_WiFi/10)/1000  # convert to W/Hz
        f = 2.4*1000000000 # carrier frequency, 2.4 GHz
        # 20 dB loss for concreate wall attenuation
        L_FS = 20*math.log10(d_LOS) + 20*math.log10(f) + 20 - 147.5 # free space loss, unit: dB
        d_BP = 3 # breakpoint distance
        if d_LOS <= d_BP:
            K = 1 # Ricean K-factor 
            X = 3 # the shadow fading before breakpoint, unit: dB        
            LargeScaleFading = L_FS + X                 
        else:
            K = 0
            X = 5 # the shadow fading after breakpoint, unit: dB                 
            LargeScaleFading = L_FS + 35*math.log10(d_LOS/d_BP) + X
        H_WiFi = math.sqrt(K/(K+1))*(math.cos(radiation_angle) + 1j*math.sin(radiation_angle)) + math.sqrt(1/(K+1))*(1/math.sqrt(2)*np.random.rand(1) + 1j/math.sqrt(2)*np.random.rand(1)) # WiFi channel transfer function                      
        channel =  (abs(H_WiFi))**2 * 10**( -LargeScaleFading / 10 ) # WiFi channel gain   
        SNR = P_WiFi*channel/(N_WiFi*B_WiFi) # range of (1000, 100000000)
    return SNR

def Capacity_NLOS(x_AP, y_AP, z_AP, x_UE, y_UE, X_length, Y_length, Z_height):
    # input x-y-z coordinat of APs to return channel gain H of NLOS
    Phi = (math.pi)/3 # semiangle: radian
    FOV = 80/180*math.pi # FOV: radian
    m = -1/(math.log2(math.cos(Phi))) # Lambertian order
    A = 0.0001 # Detector area: m**2
    n = 1.5 # Reflective index of concentrator
    UE = [x_UE, y_UE, 0] # UE Location
    AP = [x_AP, y_AP, z_AP]
    rho = 0.8 # reflection coefficient of room walls
    step = 0.1   # <--- change from 0.2 to 0.1
    Nx = int(X_length/step)
    Ny = int(Y_length/step)
    Nz = int(Z_height/step) # number of grid in each surface
    X = np.linspace(0, X_length, Nx+1)
    Y = np.linspace(0, Y_length, Ny+1)
    Z = np.linspace(0, Z_height, Nz+1)
    dA = 0.01 # reflective area of wall
    H_NLOS_W1 = [[0]*Nz]*Nx
    H_NLOS_W2 = [[0]*Nz]*Ny
    H_NLOS_W3 = [[0]*Nz]*Nx
    H_NLOS_W4 = [[0]*Nz]*Ny
    for i in range (len(X)-1):
        W1_list = []
        W2_list = []
        W3_list = []
        W4_list = []
        for j in range(len(Z)-1):
            # H11_NLOS of Wall 1 (Left), 1st reflection channel gain between AP1 and UE
            Refl_point_W1 = [0, (Y[i]+Y[i+1])/2, (Z[j]+Z[j+1])/2]
            d1 = distance(AP, Refl_point_W1)
            d2 = distance(UE, Refl_point_W1)
            # d1 = sqrt((AP(1) - Refl_point_W1(1))**2 + (AP[2] - Refl_point_W1[2])**2 + (AP(3) - Refl_point_W1(3))**2); 
            # d2 = sqrt((UE(1) - Refl_point_W1(1))**2 + (UE[2] - Refl_point_W1[2])**2 + (UE(3) - Refl_point_W1(3))**2); % distance calculation in 3-D space
            cos_phi = abs(Refl_point_W1[2] - AP[2])/d1;
            cos_alpha = abs(AP[0] - Refl_point_W1[0])/d1
            cos_beta = abs(UE[0] - Refl_point_W1[0])/d2
            cos_psi = abs(UE[2] - Refl_point_W1[2])/d2 # /sai/
            if abs(math.acos(cos_phi)) <= Phi:
                if abs(math.acos(cos_psi)) <= FOV:
                    h = (m+1)*A*rho*dA*cos_phi**m*cos_alpha*cos_beta*cos_psi*n**2/(2*math.pi**2*d1**2*d2**2*(math.sin(FOV))**2)
                else:
                    h = 0      
            else:
                h = 0
            W1_list.append(h)
            
            # H11_NLOS of Wall 2 (Front)
            Refl_point_W2 = [(X[i]+X[i+1])/2, 0, (Z[j]+Z[j+1])/2]
            d1 = distance(AP, Refl_point_W2)
            d2 = distance(UE, Refl_point_W2) 
            # d1 = sqrt((AP(1)-Refl_point_W2(1))**2 + (AP[2]-Refl_point_W2[2])**2 + (AP(3)-Refl_point_W2(3))**2); 
            # d2 = sqrt((UE(1)-Refl_point_W2(1))**2 + (UE[2]-Refl_point_W2[2])**2 + (UE(3)-Refl_point_W2(3))**2); % distance calculation in 3-D space
            cos_phi = abs(Refl_point_W2[2]-AP[2])/d1
            cos_alpha = abs(AP[0]-Refl_point_W2[0])/d1
            cos_beta = abs(UE[0]-Refl_point_W2[0])/d2
            cos_psi = abs(UE[2]-Refl_point_W2[2])/d2 # /sai/
            if abs(math.acos(cos_phi)) <= Phi:
                if abs(math.acos(cos_psi)) <= FOV:
                    h = (m+1)*A*rho*dA*cos_phi**m*cos_alpha*cos_beta*cos_psi*n**2/(2*math.pi**2*d1**2*d2**2*(math.sin(FOV))**2)
                else:
                    h = 0
            else:
                h = 0
            W2_list.append(h)
                    
            # H11_NLOS of Wall 3 (Right)
            Refl_point_W3 = [X_length, (Y[i]+Y[i+1])/2, (Z[j]+Z[j+1])/2]
            d1 = distance(AP, Refl_point_W3)
            d2 = distance(UE, Refl_point_W3)
            # d1 = sqrt((AP(1)-Refl_point_W3(1))**2 + (AP[2]-Refl_point_W3[2])**2 + (AP(3)-Refl_point_W3(3))**2); 
            # d2 = sqrt((UE(1)-Refl_point_W3(1))**2 + (UE[2]-Refl_point_W3[2])**2 + (UE(3)-Refl_point_W3(3))**2); % distance calculation in 3-D space
            cos_phi = abs(Refl_point_W3[2]-AP[2])/d1
            cos_alpha = abs(AP[0]-Refl_point_W3[0])/d1
            cos_beta = abs(UE[0]-Refl_point_W3[0])/d2
            cos_psi = abs(UE[2]-Refl_point_W3[2])/d2 # /sai/
            if abs(math.acos(cos_phi)) <= Phi:
                if abs(math.acos(cos_psi)) <= FOV:
                    h = (m+1)*A*rho*dA*cos_phi**m*cos_alpha*cos_beta*cos_psi*n**2/(2*math.pi**2*d1**2*d2**2*(math.sin(FOV))**2)
                else:
                    h = 0
            else:
                h = 0 
            W3_list.append(h)

            # H11_NLOS of Wall 4 (Back)
            Refl_point_W4 = [(X[i]+X[i+1])/2, Y_length, (Z[j]+Z[j+1])/2]
            d1 = distance(AP, Refl_point_W4)
            d2 = distance(UE, Refl_point_W4)
            # d1 = sqrt((AP(1)-Refl_point_W4(1))**2 + (AP[2]-Refl_point_W4[2])**2 + (AP(3)-Refl_point_W4(3))**2); 
            # d2 = sqrt((UE(1)-Refl_point_W4(1))**2 + (UE[2]-Refl_point_W4[2])**2 + (UE(3)-Refl_point_W4(3))**2); % distance calculation in 3-D space
            cos_phi = abs(Refl_point_W4[2]-AP[2])/d1
            cos_alpha = abs(AP[0]-Refl_point_W4[0])/d1
            cos_beta = abs(UE[0]-Refl_point_W4[0])/d2
            cos_psi = abs(UE[2]-Refl_point_W4[2])/d2 # /sai/
            if abs(math.acos(cos_phi))<= Phi:
                if abs(math.acos(cos_psi))<= FOV:
                    h = (m+1)*A*rho*dA*cos_phi**m*cos_alpha*cos_beta*cos_psi*n**2/(2*math.pi**2*d1**2*d2**2*(math.sin(FOV))**2)
                else:
                    h = 0    
            else:
                h = 0
            W4_list.append(h)
            
        H_NLOS_W1[i] = W1_list
        H_NLOS_W2[i] = W2_list
        H_NLOS_W3[i] = W3_list
        H_NLOS_W4[i] = W4_list
        
    H_NLOS = np.sum(H_NLOS_W1) + np.sum(H_NLOS_W2) + np.sum(H_NLOS_W3) + np.sum(H_NLOS_W4)
    return H_NLOS
