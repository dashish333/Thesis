#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:47:31 2020

@author: ashishdwivedi
"""
from datetime import time
import numpy as np
from l1solver import l1RegressionSolver

# SET THE BELOW FIELDS ACCORDING TO YOUR EXPERIEMNT SETTING
sketch_size = []
l1normOpt = []
n = 0 # the number of rows in put feature matrix
row = n
iterations = 10
def ON(A,b):
    print("-----ON-----")
    l1norm=[] #for storing l1 norm across various size changes
    iter_data=[]
    time_per_k = []
    u,e,vt = np.linalg.svd(A,full_matrices=False)
    leveragescore = np.sum(np.abs(u),axis=1) # for rows the matrix considered is column :
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