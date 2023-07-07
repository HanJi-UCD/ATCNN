#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 02:16:03 2022
Updated on Mon June 05, 2023

@author: qiang, Han
"""

from ATCNN_model import ATCNN
import os
import torch
import torch.nn as nn
from datetime import datetime
from utils import Csvloss, loader, get_args
import numpy as np
import warnings

warnings.filterwarnings("ignore")
args = get_args()

trail_name = '4LiFi_AP_SINR_newDesign' #
args.epochs = 51
############################################
args.lr = 0.0001 #
args.momentum = 0.95
args.weight_decay = 1e-3
args.batch_size = 256 # 256 samples for one batch in each epoch
args.test_batch_size = 256 ###
############################################
args.acc_freq = 1000 # acc 
args.log_freq = 1000 # loss
############################################
AP_size = 5 # number of WiFi+LiFi APs
ue = 25 # max supporting UE number of ATCNN
user_dim = AP_size + 1 # 
output_dim = AP_size  # 
cond_dim = (AP_size+1)*(ue) #
############################################
SNR_max = 60
SNR_min = 0
R_max = 1000 # Mbps
R_min = 1 #
############################################
save_folder = './result'
exp_folder = os.path.join(save_folder, trail_name)

if not os.path.exists(exp_folder):
    os.mkdir(exp_folder)
print('Exporting the model and log to the folder:', exp_folder)

csv = Csvloss()

# load the dataset
train_dataset = loader(input_path='dataset/4LiFi_AP_SINR/input_train.h5',
                        output_path='dataset/4LiFi_AP_SINR/output_train.h5',
                        batch_size=args.batch_size,
                        shuffle = True
                        )

test_dataset = loader(input_path='dataset/4LiFi_AP_SINR/input_test.h5',
                        output_path='dataset/4LiFi_AP_SINR/output_test.h5',
                        batch_size=args.test_batch_size,
                        shuffle = True
                        )

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#Create the model
model = ATCNN(input_dim=user_dim, cond_dim=cond_dim, cond_out_dim=user_dim, output_dim=output_dim)
model = model.to(device)

# Loss function
# criterion = nn.MSELoss().to(device)  
criterion = nn.BCELoss().to(device) # use binary cross-entropy

# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) #
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #

with open(f'{exp_folder}/config.txt','w') as f:
            f.write('Hyper parameters:\n')
            f.write(f'lr : {args.lr},batch_size:{args.batch_size}, loss_function:str(criterion), \
                    momentun : {args.momentum}, weight_decay : {args.weight_decay}, optimizor:str(optimizer)')
            f.write('\n')
            f.write('Model architecture:')
            f.write(str(model))

np.set_printoptions(suppress=True)
# Start training
count = 0
L_train = 0
c_train = 0

for epoch in range(args.epochs):
    for idx,[raw_dataset, raw_label] in enumerate(train_dataset):
        for UE_num in range(1):
            model.train() # It indicates that the model should be prepared for training and enables specific behaviors that are relevant during the training process
            
            raw_dataset = torch.tensor(raw_dataset).to(torch.float32)
            raw_label = torch.tensor(raw_label).to(torch.float32)
            
            if torch.cuda.is_available():
                raw_dataset = raw_dataset.cuda(non_blocking=True).to(torch.float32)
                raw_label = raw_label.cuda(non_blocking=True).to(torch.float32)
            
            # choose i-th UE for tarining
            Target = raw_dataset[..., 0+UE_num*user_dim:(UE_num+1)*user_dim] # choose first UE input data
            # Condition = raw_dataset[..., user_dim:] # condition without target
            Condition1 = raw_dataset[..., 0:ue*user_dim] # condition with target
            
            Condition2 = raw_dataset[..., ue*user_dim:(2*ue-1)*user_dim] # condition with target
            Condition2 = np.concatenate((Target, Condition2), axis=1)
            Condition2 = torch.tensor(Condition2).to(torch.float32)
            
            label_sub = raw_label[..., 0+UE_num*output_dim:(UE_num+1)*output_dim] # choose first UE output data
            
            for index in range(2):
                if index == 0:
                    Condition = Condition1
                else:
                    Condition = Condition2
            
                opt = model(Target, Condition) # ATCNN model 
        
                loss = criterion(opt, label_sub)
                
                L_train += loss.item()
                c_train += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if count % args.log_freq == 0:
                    log = [epoch,count]
                    print('-'*70)
                    print(datetime.now()) 
                    print('Epoch:',epoch)
                    print('Step:',f'{idx + 1}/{len(train_dataset)}')
                    print('Batch_loss:',L_train/c_train)
                    log.append(L_train/c_train)
                    L_train = 0
                    c_train = 0
                    
                    with torch.no_grad(): 
                        model.eval()
                        print('\n')
                        print('Start Evaluating')
                        L_test = 0
                        c_test = 0
                        for idx_test,[raw_dataset, raw_label] in enumerate(test_dataset):
                            
                            raw_dataset = torch.tensor(raw_dataset).to(torch.float32)
                            raw_label = torch.tensor(raw_label).to(torch.float32)
                            
                            if torch.cuda.is_available():
                                raw_dataset = raw_dataset.cuda(non_blocking=True).to(torch.float32)
                                raw_label = raw_label.cuda(non_blocking=True).to(torch.float32)
                            
                            Target = raw_dataset[..., 0+UE_num*user_dim:(UE_num+1)*user_dim] # choose first UE input data
                            # Condition = raw_dataset[..., user_dim:] # condition without target
                        
                            Condition = raw_dataset[..., 0:ue*user_dim] # condition with target

                            label_sub = raw_label[..., 0+UE_num*output_dim:(UE_num+1)*output_dim] # choose first UE output data
                            
                            opt = model(Target, Condition)
                            loss = criterion(opt, label_sub)
                            
                            L_test += loss.item()
                            c_test += 1
                        print('Eval_loss:',L_test / c_test)
                        log.append(L_test / c_test) 
                        
                        csv.update(log, f'{exp_folder}/log.csv')
                
        count += 1
        
        # if not os.path.exists('trained_model/4LiFi_AP_newArrange4_new'):
        #     os.mkdir('trained_model/4LiFi_AP_newArrange4_new')
        # print('*********************** Saving ATCNN Model *************************')
        # torch.save(model.state_dict(), 'trained_model/4LiFi_AP_newArrange4_new/4LiFi_AP_newArrange4_epoch%s.pth'%str(epoch))   

torch.save(model.state_dict(), 'trained_model/4LiFi_AP_SINR_newDesign.pth')
        
        
        
        
        
        