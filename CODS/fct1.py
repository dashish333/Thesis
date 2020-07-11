#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 02:31:08 2020

@author: ashishdwivedi
"""


import numpy as np
from l1solver import l1RegressionSolver
import copy
from datetime import time
from scipy.linalg import block_diag,hadamard

# SET THE BELOW FIELDS ACCORDING THE YOUR EXPERIEMNT SETTING
sketch_size = []
l1normOpt = []
n = 0 # the number of rows in put feature matrix
row = n
iterations = 10
#####################################################



## ------------------------- Main Function --------------------
def FCT(A,b):
    print("-------- FCT -------")
    l1norm=[] #for storing l1 norm across various size changes
    iter_data=[]
    time_per_k = []
    CT = getCT()
    Q,R=np.linalg.qr(np.dot(CT,A))
    u = np.dot(A,np.linalg.pinv(R))
#     print("u shape is:",u.shape)
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



###----------------------------- Util Functions ----------------
## BCH construction
def getB(r1):
    B = np.zeros((r1,2*n))
    for i in range(2*n):
        choice = np.random.randint(0,high = r1)
        B[choice][i] = 1
#     print("B shape",B.shape)
    return B
### ---------------------
def getC():
    C = [np.random.standard_cauchy()  for i in range(2*n)]
#     print("C shape",len(C))
    return C
## -----------------------
def getH(s):
    n_by_s = int(n/s)
#     print(n_by_s)
    Is = np.eye(s)
    Hs = np.multiply(1/np.sqrt(s),hadamard(s))
    Gs = np.vstack([Hs,Is])
    H_bar = block_diag(*([Gs]*n_by_s))
    print("Normalized H shape",H_bar.shape)
    return H_bar
def getCT():
    # the params r1 and s are 
    # same as of discussed in paper
    # setting s to be power of 2 greater than r1
    r1 = 16
    s = 64
    C = getC()
    # fixing r1 = 8
    B = getB(r1)
    H_bar = getH(s)
    CH = np.multiply(H_bar, np.array(C)[:,np.newaxis])
    BCH = np.dot(B,CH)
    CT = np.multiply(4,BCH)
    return CT