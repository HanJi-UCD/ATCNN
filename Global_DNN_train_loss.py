#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 02:16:03 2022
Updated on Mon June 05, 2023

@author: qiang, Han
"""

from ATCNN_model import global_dnn
import os
import torch
import torch.nn as nn
from datetime import datetime
from utils import Csvloss, loader, get_args
import numpy as np
import warnings

warnings.filterwarnings("ignore")
args = get_args()

trail_name = 'DNN_4LiFi_45UE_Final' #
args.epochs = 51
############################################
args.lr = 0.001 #
args.momentum = 0.95
args.weight_decay = 1e-3
args.batch_size = 256 # 256 samples for one batch in each epoch
args.test_batch_size = 256 ###
############################################
args.acc_freq = 100 # acc 
args.log_freq = 100 # loss
############################################
AP_size = 5 # number of WiFi+LiFi APs
ue = 45 # max supporting UE number
input_dim = (AP_size+1)*ue
output_dim = AP_size*ue # 
############################################
save_folder = './result'
exp_folder = os.path.join(save_folder, trail_name)

if not os.path.exists(exp_folder):
    os.mkdir(exp_folder)
print('Exporting the model and log to the folder:', exp_folder)

csv = Csvloss()

# load the dataset
train_dataset = loader(input_path='dataset/GT_4LiFi_3W_DNN/45UE/input_train.h5',
                        output_path='dataset/GT_4LiFi_3W_DNN/45UE/output_train.h5',
                        batch_size=args.batch_size,
                        shuffle = True
                        )

test_dataset = loader(input_path='dataset/GT_4LiFi_3W_DNN/45UE/input_test.h5',
                        output_path='dataset/GT_4LiFi_3W_DNN/45UE/output_test.h5',
                        batch_size=args.test_batch_size,
                        shuffle = True
                        )

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#Create the model
model = global_dnn(input_dim=input_dim, output_dim=output_dim)
model = model.to(device)

# Loss function
# criterion = nn.MSELoss().to(device)  
# criterion = nn.BCELoss().to(device) # use binary cross-entropy
criterion = nn.CrossEntropyLoss().to(device) # use cross-entropy

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
    for idx,[ipt, label] in enumerate(train_dataset):
        model.train()
        
        ipt = torch.tensor(ipt).to(torch.float32)
        label = torch.tensor(label).to(torch.float32)
        
        if torch.cuda.is_available():
            ipt = ipt.cuda(non_blocking=True).to(torch.float32)
            label = label.cuda(non_blocking=True).to(torch.float32)

        opt = model(ipt)
        loss = criterion(opt, label)
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
                for idx_test,[ipt,label] in enumerate(test_dataset):
                    ipt = torch.tensor(ipt).to(torch.float32)
                    label = torch.tensor(label).to(torch.float32)
                    if torch.cuda.is_available():
                        ipt = ipt.cuda(non_blocking=True).to(torch.float32)
                        label = label.cuda(non_blocking=True).to(torch.float32)
                    opt = model(ipt)
                    loss = criterion(opt, label)
                    L_test += loss.item()
                    c_test += 1
                print('Eval_loss:',L_test / c_test)
                log.append(L_test / c_test)
                csv.update(log, f'{exp_folder}/log.csv')
        count += 1 

torch.save(model.state_dict(), 'trained_model/Final/DNN_4LiFi_45UE_CE.pth')
        
        
        
        
        
        