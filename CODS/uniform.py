#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 02:28:46 2020

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

def USS(A,b):
    print("--USS--")
    l1norm = []
    ratio=[]
    iter_data=[]
    time_per_k=[]
    for k in sketch_size:
        r=0
        temp_data=[]
        temp_time = 0
       # print("---- Sample Size ----",k)
        for i in range(iterations):
            index = np.random.choice(row,k,replace=False)
            A_sketch = A[index,:]
            b_sketch = b[index]
            start = time.time()
            x_tilde = np.array(l1RegressionSolver(A_sketch,b_sketch))
            end = time.time()
            temp_time+=end-start
            regression_value = np.linalg.norm((A.dot(x_tilde)-b),ord=1)
            temp_data.append(regression_value)
            r+=regression_value
        ## for loop ends:
        time_per_k.append(temp_time/iterations)
        iter_data.append(temp_data)
        r/=iterations
        l1norm.append(r)
    ratio=np.array(l1norm)/np.array(l1normOpt)
    print("l1norm_each sample size-",l1norm)
    print("ratio_each sample size-",ratio)
    return ratio,l1norm,time_per_k