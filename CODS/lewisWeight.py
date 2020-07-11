#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 02:29:56 2020

@author: ashishdwivedi
"""

import numpy as np
from l1solver import l1RegressionSolver
import copy
from datetime import time

# SET THE BELOW FIELDS ACCORDING THE YOUR EXPERIEMNT SETTING
sketch_size = []
l1normOpt = []
n = 0 # the number of rows in put feature matrix
row = n
iterations = 10
#####################################################


def LewisIterate(A,w):
    # beta = 1
    # p = 1
    global n
    w_hat_iter = []
    W = np.diag(w)
    W_inv = np.linalg.inv(W) # p = 1 => 1 - 2/p = -1
    T = np.linalg.inv(np.matmul(np.matmul(A.T,W_inv),A))
    w_hat = np.sqrt(np.diagonal(np.matmul(np.matmul(A,T),A.T)))
    return w_hat

#################
####
#################

def ApproxLewisWeight(A,T):
    global n
    wi = 1
    w  = list(np.ones((n)))
    for t in range(1,T):
        w = LewisIterate(A,w)
        print(w)
    return w
## ---------------------------
def lewisWeight(A,b):
    print("---------- Lewis Weight ------------")
    l1norm=[] #for storing l1 norm across various size changes
    iter_data=[]
    time_per_k = []
    
    #u,e,vt = np.linalg.svd(A,full_matrices=False)
    #leveragescore = np.sum(np.abs(u),axis=1) # for rows the matrix considered is column :
    
    # T = 2
    w = ApproxLewisWeight(A,2)
    leveragescore = np.array(w)
    prob = leveragescore/sum(leveragescore)  
    for k in sketch_size: # s represent the list of number of cluster
        r=0
        temp_data=[]
        temp_time = 0
        for i in range(iterations):
            index = np.random.choice(row,k,replace = False,p=prob)
            A_sketch = A[index,:]
            b_sketch = b[index]
            index_prob = prob[index][:,np.newaxis]
            min_prob = np.min(index_prob)
            index_prob = min_prob * (1/index_prob)
            #print(type(index_prob),index_prob.shape)
            A_sketch = np.multiply(A_sketch,index_prob,dtype=float)
            b_sketch = np.multiply(b_sketch,index_prob,dtype=float)
            start = time.time()
            x_tilde = np.array(l1RegressionSolver(A_sketch,b_sketch))
            end = time.time()
            temp_time += (end - start) # in seconds
            regression_value=np.linalg.norm((A.dot(x_tilde)-b),ord=1)
            temp_data.append(regression_value)
            r+=regression_value
        time_per_k.append(temp_time/iterations)    
        iter_data.append(temp_data)
        r/=iterations
        l1norm.append(r)
    ratio = np.array(l1norm)/l1normOpt
    print("")
    print("L1Norm = ",l1norm)
    print("Ratio = ",ratio)
    print("Time(secs) = ",time_per_k)
    print("---------------------\n")
    return ratio,l1norm,time_per_k