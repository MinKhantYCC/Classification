# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 17:21:54 2022

@author: Xero
"""
import pandas as pd
import numpy as np
from sigmoid import sigmoid 

def predict(X,theta):
    '''
    Description
    -----------
    This function predict y value for binary classification based on trained theta values and X.

    Parameters
    ----------
    X : numpy matrix of float
        input features data in shape of (#of samples * #of features)
    theta : numpy matrix of float
        input weight in shape of (#of features * #of class)
    
    Return
    ------
    y_pred : numpy matrix of float
        predicted y value in shape of (#of samples * 1)
    '''
    m,n = X.shape
    k = theta.shape[1]
    y = np.zeros((m,k))
    
    y = sigmoid(X,theta)
    for i in range(m):
        if y[i] < 0.5:
            y[i] = 0
        else:
            y[i] = 1
    return y