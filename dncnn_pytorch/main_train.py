# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1. 
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset
from dncnn import DnCNN
from utils import sum_squared_error, log, findLastCheckpoint
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
n_epoch = args.epoch
sigma = args.sigma

save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def train(model, device, train_loader, loss_fn, optimizer, epoch): 
    model.train()
    
    running_train_loss = 0
    for step, data in enumerate(train_loader):
        optimizer.zero_grad()
        noisy_image, target_image = data[0].to(device), data[1].to(device)
        residual_image = model(noisy_image)
        clean_image = noisy_image - residual_image 
        clean_image = clean_image.to(device)
        loss = loss_fn(target_image, clean_image) / 2 
        running_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            log('%4d %4d / %4d loss = %2.4f' % (epoch+1, step, xs.size(0)//batch_size, loss.item()/batch_size))
    
    log('epoch = %4d , loss = %4.4f' % (epoch+1, running_train_loss/len(train_loader)))
     
    return running_train_loss

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('===> Building model')
    model = DnCNN()
    model = model.to(device)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  
    
    for epoch in range(n_epoch):
        
        xs = dg.datagenerator(data_dir=args.train_data)
        xs = xs.astype('float32')/255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, N*C*H*W
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, num_workers=2, drop_last=True, batch_size=batch_size, shuffle=True)
        
        loss = train(model, device, DLoader, criterion, optimizer, epoch)  
        scheduler.step(epoch)
      
        torch.save(model, os.path.join(save_dir, 'model_%03d_training_mod.pth' % (epoch+1)))
        
    torch.save(model, os.path.join(save_dir, 'final_model.pth'))







