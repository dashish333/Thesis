#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 02:25:56 2020

@author: ashishdwivedi
"""
import numpy as np
from l1solver import l1RegressionSolver
import copy

# SET THE BELOW FIELDS ACCORDING THE YOUR EXPERIEMNT SETTING
sketch_size = []
l1normOpt = []
n = 0 # the number of rows in put feature matrix
row = n
iterations = 10
#####################################################

def RIVS(A,b):
    print("\n -- RIVS --")
    l1norm = np.zeros(len(sketch_size))
    ratio=[]
    iter_data=[]
    time_per_k=[]
    X = copy.deepcopy(A)
    Zp = np.linalg.inv(np.matmul(X.T,X))
    H = 1-np.diagonal((np.matmul(np.matmul(X,Zp),X.T)))
    H = [(item > 0) * item for item in H]
    hi = H
    for i in range(iterations):
        print("Iteration ------------------------- = ",i)
        listS = revVBS(copy.deepcopy(Zp),X,copy.deepcopy(hi))
        for pos in range(len(sketch_size)):
            ## the index store the coreset size in decreasing order
            fetchIndexes = listS[pos]
            A_sketch = A[fetchIndexes,:]
            b_sketch = b[fetchIndexes]
            x_tilde = np.array(l1RegressionSolver(A_sketch,b_sketch))
            regression_value = np.linalg.norm((A.dot(x_tilde)-b),ord=1)
            l1norm[pos] +=regression_value
    l1norm = l1norm/iterations
    ratio=np.array(l1norm)/np.array(l1normOpt)
    print("")
    print("L1Norm = ",l1norm[::-1])
    print("Ratio = ",ratio[::-1])
    print("---------------------\n")
    return ratio[::-1],l1norm[::-1]

## ------------------------------------
def revVBS(Z,X,h):
    listSos = []
    S = np.arange(n)
    revS = sketch_size[::-1] 
    cS = 0 # current Sample Index
    k = revS[cS] # highest corset size
    lenS = len(revS)
    while S.size > k:
        prob = h/np.sum(h)
        if(np.isnan(prob).any()):
            print("Failure = ",np.sum(h),len(h))
        value = np.random.choice(S,size=None,p=prob)
        index = int(np.where(S==value)[0])
        Zxi = np.matmul(Z,X[value,:][:,np.newaxis])
        v = Zxi / np.sqrt(h[index])        
        S = np.delete(S,index)
        h = np.delete(h,index)
        Xv = np.matmul(X[S,:],v)
        h = np.subtract(h,Xv.flatten()**2)
        h = [(item > 0) * item for item in h]
        Z += np.matmul(v,v.T)
        if(S.size == k):
            cS +=1
            if cS < lenS:
                print("RIVS k completed = ", k)
                k = revS[cS]
                #print("current Samples required = ",k)
            listSos.append(list(S))
    del Z,X,h
    return listSos