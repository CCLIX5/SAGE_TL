import threading
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
import matplotlib.cm as cm
from sklearn import datasets
from math import *
from argparse import ArgumentParser
import seaborn as sns; sns.set_style('white')
from backpack import extend, backpack, extensions
from torch.distributions.multivariate_normal import MultivariateNormal
import tikzplotlib
from matplotlib import rc
import pickle
import csv
import pandas as pd
import sage
from uLSIF_code import uLSIF
from torch.nn import BCELoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

rc("text", usetex=False)

#NN model
class Model(nn.Module):

    def __init__(self, n, h):
        super(Model, self).__init__()

        self.feature_extr = nn.Sequential(
            nn.Linear(n, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.feature_extr(x)
        return x


def source_train(X, Y, model, opt):
    X_train = X
    y_train = Y.unsqueeze(1)
    model.train()
    criterion = BCELoss()
    for it in range(500):
        y = model(X_train)
        l = criterion(y, y_train)
        l.backward()
        opt.step()
        opt.zero_grad()
    print(f'Loss in train dataset: {l.item():.3f}')
    return model

def test_model(X,Y,model):
    X_test = X
    y_test = Y.unsqueeze(1)
    y_predict = model(X_test)
    criterion = BCELoss()
    l = criterion(y_predict, y_test)
    print(f'Loss in test dataset: {l.item():.3f}')
    #AUC
    y_te_nump = y_test.detach().numpy()
    y_pr_nump = y_predict.detach().numpy()
    auc_ = roc_auc_score(y_te_nump, y_pr_nump)
    print(f'AUC for test dataset: {auc_.item():.3f}')
    return 0

def TL(X, Y, model, opt):

    X_train = X
    y_train = Y.unsqueeze(1)
    model.train()
    criterion = BCELoss()
    for it in range(100):
        y = model(X_train)
        l = criterion(y, y_train)
        opt.zero_grad()
        l.backward()
        opt.step()
    return model

def select_topn(loc_path,dataname,n,rank,kword=''):
    data_old = pd.read_csv(loc_path+dataname)
    data_ = data_old.copy()
    mylist = data_.columns
    feature_names = [mylist[x] for x in range(1,len(mylist)-2)] 
    feature_chose = []
    for i in range(n):
        j = rank[i]
        feature_chose = feature_chose + [feature_names[j]]
    data_new = data_[feature_chose]
    index_name = dataname.find('.csv')
    new_name = dataname[:index_name]+'_new'+kword+dataname[index_name:]
    data_new.to_csv(loc_path+new_name)
    print('finish selection!')
    return feature_chose

def train_model(lr, momentum, weight_decay):
    print('SOURCE OUTPUT:', lr, momentum, weight_decay)
    model_n = Model(n=N_new, h=H)
    opt = optim.SGD(model_n.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model_n = source_train(X_tr_n_s_tensor, y_tr_n_s_tensor, model_n, opt)
    model_n.eval()
    test_model(X_te_n_s_tensor, y_te_n_s_tensor, model_n)

def train_with_output_file(lr, momentum, weight_decay):
    output_filename = 'output_lr' + str(lr) + '_m' + str(momentum) + '_wd' + str(weight_decay) + '.txt'
    with open(os.path.join('output', output_filename), 'w') as out_file:
        sys.stdout = out_file
        train_model(lr, momentum, weight_decay)
        sys.stdout = sys.__stdout__



if __name__ == '__main__':
    parser = ArgumentParser(description='mimic')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--momentum', type=float, required=True)
    parser.add_argument('--wd', type=float, required=True)
    args = parser.parse_args()

    np.random.seed(777)
    torch.manual_seed(99999)

    #load data
    loc_path = 'mimic'
    data1 = pd.read_csv(loc_path +'/Source.csv')
    data2 = pd.read_csv(loc_path +'/Target.csv')
    source_data = data1.copy()
    target_data = data2.copy()
    mylist = data1.columns
    feature_names = [mylist[x] for x in range(1,len(mylist)-2)]
    source_x = source_data.values[:, 1:len(mylist)-2]
    source_y = source_data.values[:, -2]
    target_x = target_data.values[:, 1:len(mylist)-2]
    target_y = target_data.values[:, -2]

    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(source_x,source_y,test_size=0.3, random_state=42)
    X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(target_x,target_y,test_size=0.3, random_state=42)

    X_tr_s = np.float64(X_tr_s)
    y_tr_s = np.float64(y_tr_s)
    X_tr_t = np.float64(X_tr_t)
    y_tr_t = np.float64(y_tr_t)

    X_tr_s_tensor = torch.from_numpy(X_tr_s).float()
    y_tr_s_tensor = torch.from_numpy(y_tr_s).float()

    X_tr_t_tensor = torch.from_numpy(X_tr_t).float()
    y_tr_t_tensor = torch.from_numpy(y_tr_t).float()

    print('---------------------Train source model total rank---------------------')
    # # source training
    M, N = X_tr_s.shape
    H = 20  # num. hidden units
    WEIGHT_DECAY = 3e-3
    LR = 1e-3
    MOMENTUM = 0.9
#----------------------------------------------for i in [], pick i do prediction
    for TOP_N in [30]:
        print("selected top ",TOP_N,"features---------------------------")
        print("source choose")
        data3 = source_data.values[:, 1:len(mylist)-2]
        source_new_x = data3.copy()
        X_tr_n_s, X_te_n_s, y_tr_n_s, y_te_n_s = train_test_split(source_new_x,source_y,test_size=0.3, random_state=42)
        X_tr_n_s = np.float64(X_tr_n_s)
        X_te_n_s = np.float64(X_te_n_s)
        y_tr_n_s = np.float64(y_tr_n_s)
        y_te_n_s = np.float64(y_te_n_s)
        X_tr_n_s_tensor = torch.from_numpy(X_tr_n_s).float()
        X_te_n_s_tensor = torch.from_numpy(X_te_n_s).float()
        y_tr_n_s_tensor = torch.from_numpy(y_tr_n_s).float()
        y_te_n_s_tensor = torch.from_numpy(y_te_n_s).float()
        M_new, N_new = X_tr_n_s.shape
    

        print("LR is ", LR, ":")
        print("source:")
        model_n = Model(n=N_new, h=H)
        opt = optim.SGD(model_n.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        model_n = source_train(X_tr_n_s_tensor, y_tr_n_s_tensor, model_n, opt)
        model_n.eval()
        test_model(X_te_n_s_tensor, y_te_n_s_tensor, model_n)

    