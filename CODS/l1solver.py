#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 02:18:23 2020

@author: ashishdwivedi
"""
from l1 import l1
from cvxopt import matrix

def l1RegressionSolver(A,b):
    def l1_regression(A,b):
        A = matrix(A) # converitng to matrix, format accepted by the solver
        b = matrix(b)
        return l1(A,b)