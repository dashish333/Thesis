#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 02:27:18 2020

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


def creatCountSketch(s):
    S = np.zeros((s,n))
    D = [1 for i in range(n)]
    for i in range(n):
        j = np.random.choice(s)
        S[j][i] = np.random.choice([1,-1])
        random_value = np.random.exponential()
        D[i] = 1/random_value
    SD = np.multiply(S,np.array(D))
    return SD
## ---------------------- MAIN Function --------------------
def invExpCountSketch(A,b):
    print("------- Count Sketch Inv Exp ---------")
    l1norm=[] #for storing l1 norm across various size changes
    iter_data=[]
    time_per_k = []
    for k in sketch_size: # s represent the list of number of cluster
        r=0
        temp_data=[]
        temp_time = 0
        for i in range(iterations):
            SD = creatCountSketch(k)
            A_sketch = np.matmul(SD,A)
            b_sketch = np.matmul(SD,b)
            start = time.time()
            x_tilde = np.array(l1RegressionSolver(A_sketch,b_sketch))
            end = time.time()
            temp_time += (end - start) # in seconds
            regression_value=np.linalg.norm((A.dot(x_tilde)-b),ord=1)
            temp_data.append(regression_value)
            r+=regression_value
        time_per_k.append(temp_time/iterations)    
        iter_data.append(temp_data)
        r/=10
        l1norm.append(r)
    ratio = np.array(l1norm)/l1normOpt
    print("")
    print("L1Norm = ",l1norm)
    print("Ratio = ",ratio)
    print("Time(secs) = ",time_per_k)
    print("---------------------\n")
    return ratio,l1norm,time_per_k