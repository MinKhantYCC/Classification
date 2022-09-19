# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:09:29 2022

@author: Xero
"""
import pandas as pd
import numpy as np

def sigmoid(X,theta):
    '''
    Parameters
    ----------
    X : array like input
        input data with (mxn).
    theta : array like input weight
        weight to calculate y (nxk).

    Returns
    -------
    y : array-like output
        predicted value by x and theta (mxk).

    '''
    m,n = X.shape
    y = np.zeros((m,1))
    hyp = (-1)*np.dot(X,theta)
    y = 1/(1+np.exp(hyp))
    return y