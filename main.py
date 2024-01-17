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
import csv
import pandas as pd
import sage
from uLSIF_code import uLSIF
from torch.nn import BCELoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

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
    print(f'New Loss in tarining set after TL: {l.item():.3f}')

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

if __name__ == '__main__':
    parser = ArgumentParser(description='mimic')
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


    M, N = X_tr_s.shape
    H = 20  # num. hidden units
    WEIGHT_DECAY = 0
    LR = 0.01
    MOMENTUM = 0.9
    print("---------------------Train target local total rank---------------------")
    # target local training
    M_t, N_t = X_tr_t.shape
    model_t = Model(n=N_t, h=H) 
    opt = optim.SGD(model_t.parameters(), lr=LR, momentum=MOMENTUM)
    model_t = source_train(X_tr_t_tensor, y_tr_t_tensor, model_t, opt) 
    model_t.eval()
    # Setup and calculate
    imputer_t = sage.MarginalImputer(model_t, X_tr_t[:200])
    estimator_t = sage.PermutationEstimator(imputer_t, 'mse')
    sage_values_t = estimator_t(X_tr_t, y_tr_t)
    target_local = []
    target_local.append(sage_values_t.values) #sage value based on target data
    t_LOC_abs_sage = list(map(abs,sage_values_t.values))
    t_LOC_intmed1 = sorted(range(len(t_LOC_abs_sage)), key=lambda k: t_LOC_abs_sage[k], reverse=True)
    t_LOC_intmed2 = sorted(range(len(t_LOC_intmed1)), key=lambda k: t_LOC_intmed1[k])
    t_LOC_sorted_id = list(map(lambda x:x+1,t_LOC_intmed2))
    target_local.append(t_LOC_sorted_id) #sort sage value and output index
    target_local = pd.DataFrame(columns=feature_names, data=target_local)
    target_local.to_csv(loc_path+'/NN_target_local.csv')

    # Create new dataset based on sage value
    rank2 = np.argsort(t_LOC_abs_sage)[::-1] #total rank 2

    print('---------------------TL total rank---------------------')
    # target TL training
    M_t, N_t = X_tr_t.shape
    model_t_TL = Model(n=N_t, h=H) 
    opt = optim.SGD(model_t_TL.parameters(), lr=LR, momentum=MOMENTUM)
    model_t_TL = source_train(X_tr_t_tensor, y_tr_t_tensor, model_t_TL, opt) 
    model_t_TL.eval()

    #test uLSIF
    x_de = X_tr_t.T 
    x_nu = X_tr_s.T #source  
    wh_x_de_t = uLSIF.uLSIF(x_de,x_nu)
    sum_t = wh_x_de_t.sum(axis=0)
    p_t = wh_x_de_t/sum_t

    # Setup and calculate
    imputer_t_TL = sage.MarginalImputer(model_t_TL, X_tr_t[:200])
    estimator_t_TL = sage.PermutationEstimator(imputer_t_TL, 'mse')
    sage_values_t_TL = estimator_t_TL(X_tr_t, y_tr_t,p_t)
    target_TL = []
    target_TL.append(sage_values_t_TL.values) #sage value based on target data
    t_TL_abs_sage = list(map(abs,sage_values_t_TL.values))
    t_TL_intmed1 = sorted(range(len(t_TL_abs_sage)), key=lambda k: t_TL_abs_sage[k], reverse=True)
    t_TL_intmed2 = sorted(range(len(t_TL_intmed1)), key=lambda k: t_TL_intmed1[k])
    t_TL_sorted_id = list(map(lambda x:x+1,t_TL_intmed2))
    target_TL.append(t_TL_sorted_id) #sort sage value and output index
    target_TL = pd.DataFrame(columns=feature_names, data=target_TL)
    target_TL.to_csv(loc_path+'/NN_target_TL.csv')

    # Create new dataset based on sage value
    rank3 = np.argsort(t_TL_abs_sage)[::-1] #total rank 3

#----------------------------------------------for i in [], pick i do prediction
    for TOP_N in [8,10,12,14,16]:
        print("selected top ",TOP_N,"features---------------------------")
        # Create new dataset based on sage value
        print("source choose")
        feature_pick1 = select_topn(loc_path,'/Source.csv',TOP_N,rank3)
        data3 = source_data[feature_pick1]
        source_new_x = data3.copy()  
        X_tr_n_s, X_te_n_s, y_tr_n_s, y_te_n_s = train_test_split(source_new_x,source_y,test_size=0.3, random_state=42) 
        # train new source model
        X_tr_n_s = np.float64(X_tr_n_s)
        X_te_n_s = np.float64(X_te_n_s)
        y_tr_n_s = np.float64(y_tr_n_s)
        y_te_n_s = np.float64(y_te_n_s)
        X_tr_n_s_tensor = torch.from_numpy(X_tr_n_s).float()
        X_te_n_s_tensor = torch.from_numpy(X_te_n_s).float()
        y_tr_n_s_tensor = torch.from_numpy(y_tr_n_s).float()  
        y_te_n_s_tensor = torch.from_numpy(y_te_n_s).float() 
        M_new, N_new = X_tr_n_s.shape    
    #------------------------------------------------------------------------------------------------------
        print("target local choose")
        feature_pick2 = select_topn(loc_path,'/Target.csv',TOP_N,rank2)
        data4 = target_data[feature_pick2] 
        target_new_x = data4.copy()  
        X_tr_n_t, X_te_n_t, y_tr_n_t, y_te_n_t = train_test_split(target_new_x,target_y,test_size=0.3, random_state=42) 

        # train new target model
        X_tr_n_t = np.float64(X_tr_n_t)
        X_te_n_t = np.float64(X_te_n_t)
        y_tr_n_t = np.float64(y_tr_n_t)
        y_te_n_t = np.float64(y_te_n_t)
        X_tr_n_t_tensor = torch.from_numpy(X_tr_n_t).float()
        X_te_n_t_tensor = torch.from_numpy(X_te_n_t).float()
        y_tr_n_t_tensor = torch.from_numpy(y_tr_n_t).float()  
        y_te_n_t_tensor = torch.from_numpy(y_te_n_t).float() 
        M_t_new, N_t_new = X_tr_n_t.shape    

    #--------------------------------------------------------------------------------------
        print("target TL choose")
        feature_pick_TL = select_topn(loc_path,'/Target.csv',TOP_N,rank3,'TL')
        data_TL = target_data[feature_pick_TL] 
        target_new_x_TL = data_TL.copy()  
        X_tr_n_t_TL, X_te_n_t_TL, y_tr_n_t_TL, y_te_n_t_TL = train_test_split(target_new_x_TL,target_y,test_size=0.3, random_state=42) 
        # train new target model
        X_tr_n_t_TL = np.float64(X_tr_n_t_TL)
        X_te_n_t_TL = np.float64(X_te_n_t_TL)
        y_tr_n_t_TL = np.float64(y_tr_n_t_TL)
        y_te_n_t_TL = np.float64(y_te_n_t_TL)
        X_tr_n_t_TL_tensor = torch.from_numpy(X_tr_n_t_TL).float()
        X_te_n_t_TL_tensor = torch.from_numpy(X_te_n_t_TL).float()
        y_tr_n_t_TL_tensor = torch.from_numpy(y_tr_n_t_TL).float()  
        y_te_n_t_TL_tensor = torch.from_numpy(y_te_n_t_TL).float() 
        M_t_new_TL, N_t_new_TL = X_tr_n_t_TL.shape    

        for LR in [0.01]:
            print("LR is ",LR,":")
            print("source:")
            model_n = Model(n=N_new, h=H)
            opt = optim.SGD(model_n.parameters(), lr=LR, momentum=MOMENTUM)
            model_n = source_train(X_tr_n_s_tensor, y_tr_n_s_tensor, model_n, opt) 
            model_n.eval()
            #valuate new model in test data (source domain)
            test_model(X_te_n_s_tensor,y_te_n_s_tensor,model_n)

            print("target local:")
            model_n_t = Model(n=N_t_new, h=H)
            opt = optim.SGD(model_n_t.parameters(), lr=LR, momentum=MOMENTUM)
            model_n_t = source_train(X_tr_n_t_tensor, y_tr_n_t_tensor, model_n_t, opt) 
            model_n_t.eval()
            #valuate new model in test data (target domain)
            test_model(X_te_n_t_tensor,y_te_n_t_tensor,model_n_t)

            print("Do not transfer parameter:")
            model_ntl = Model(n=N_t_new_TL, h=H) 
            opt_ntl = optim.SGD(model_ntl.parameters(), lr=LR, momentum=MOMENTUM)
            model_ntl = source_train(X_tr_n_t_TL_tensor, y_tr_n_t_TL_tensor, model_ntl, opt_ntl) 
            model_ntl.eval()
            #valuate new model in test data (target domain)
            test_model(X_te_n_t_TL_tensor,y_te_n_t_TL_tensor,model_ntl)

            print("parameter_TL:")
            opt_TL = optim.SGD(model_n.feature_extr.parameters(), lr=LR, 
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

            model_n = TL(X_tr_n_t_TL_tensor, y_tr_n_t_TL_tensor, model_n, opt_TL)

            model_n.eval()
            test_model(X_te_n_t_TL_tensor,y_te_n_t_TL_tensor,model_n)
