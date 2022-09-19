# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 13:38:41 2022

@author: Xero
"""

import pandas as pd
import numpy as np
from sigmoid import sigmoid
from cost_fun import cost

def gradient(X, y, theta, lamb = 0, alpha = 1, max_iter=100):
    '''
    Description
    -----------
    This function find the error and theta values which is the best for min error.
    To use this function, cost function and sigmoid function are required.

    Parameters
    ----------
    X : array-like (mxn)-shpae input
    y : array-like (mxk)-shape output
    theta : array-like (nxk)-shape input for weight
    lamb : float or int, optional
        Variable to calculate regularization. The default is 0.
    alpha : int or float
        learning rate parameter for gradient descent. The default is 1.
    max_iter : int
        maximum number of iteration in gradient descent. The default is 100.

    Returns
    -------
    J_hist : list of float
        errors between y_pred and actual y
    theta : np.array
        trained theta values by gradient descent

    '''
    J_hist = list()
    
    #size
    m,n = X.shape
    
    #train model for max_iter
    for i in range(max_iter):
        J = float(cost(X, y, theta,lamb))     #find cost error
        J_hist.append(J)                      #record cost error in J history list
        
        #gradient
        y_pred = sigmoid(X,theta)             #predict y value
        theta_temp = theta - (alpha/m)*(np.dot(X.T,(y_pred-y))) #find theta
        reg = alpha*(lamb/m)*theta            #regularization term
        theta_temp[1:] = theta_temp[1:]-reg[1:]  #find theta with regularization
        
        theta = theta_temp                    #save trained theta

    return J_hist, theta