#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 02:21:04 2020

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
def BRIVA(A,b,alpha):
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
        listS = batchRevVBS(copy.deepcopy(Zp),X,copy.deepcopy(hi),alpha)
        for pos in range(len(sketch_size)):
            fetchIndexes = listS[pos]
            #print("length of fI = ",len(fetchIndexes))
            A_sketch = A[fetchIndexes,:]
            b_sketch = b[fetchIndexes]
            x_tilde = np.array(l1RegressionSolver(A_sketch,b_sketch))
            regression_value = np.linalg.norm((A.dot(x_tilde)-b),ord=1)
            l1norm[pos] +=regression_value
    # these values are in decresing order of sample sizes required
    l1norm = l1norm/iterations
    ratio=np.array(l1norm)/np.array(l1normOpt)
    print("")
    print("L1Norm = ",l1norm[::-1])
    print("Ratio = ",ratio[::-1])
    print("---------------------\n")
    return ratio[::-1],l1norm[::-1]

## --------------------
def batchRevVBS(Z,X,h,alpha):
    
    # alpha is some factor of window
    stringF = ""  
    hasNeg = 0
    listSosV4 = []
    S = np.arange(n)
    revS = sketch_size[::-1] # reversing the list of sample 
    cS = 0 # current Sample Index
    k = revS[cS]
    lenS = len(revS)
    window = S.size - k
    while S.size > k:
        prob = h/np.sum(h)
        bs = max(int(alpha*window),1)
        indexes = set(np.random.choice(S,size=bs,replace=False,p=prob))
        indexDeleted = []
        for value in indexes:
            indexDeleted.append(np.asscalar(np.where(S==value)[0]))
            window -=1 
        V = X[list(indexes),:].T
        h = np.delete(h,indexDeleted)
        S = np.delete(S,indexDeleted)
        ### update to Z ----------
        ### sherman-morrison-woodburry
        ### (X^TX - VV^T)^-1 = Z + ZV(Ik - V^T Z V)^-1 V^T Z
        ZV = np.matmul(Z,V)
        VTZ = np.matmul(V.T,Z)
        VTZV = np.matmul(VTZ,V)
        Ik = np.identity(bs)
        Ik = Ik - VTZV
        Ik = np.linalg.inv(Ik)
        Z = Z + np.matmul(ZV,(np.matmul(Ik,VTZ)))
        ### Update done ####
        H = 1-np.diagonal((np.matmul(np.matmul(X[list(S)],Z),X[(list(S))].T)))
        H = [(i > 0) * i for i in H]
        h = H
        if(S.size == k):
            cS +=1
            if cS < lenS:
                k = revS[cS]
                #print("current Samples required = ",k)
                window = S.size - k
            listSosV4.append(list(S))
    del Z,X,h
    return listSosV4