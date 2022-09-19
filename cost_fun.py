# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:31:27 2022

@author: Xero
"""

import pandas as pd
import numpy as np
from sigmoid import sigmoid


def cost(X,y,theta, lamb=0):
    '''
    Description
    -----------
    This function calculates the cost error between predicted y value and actual y value, then return 
    the error. To use this function, sigmoid function is required.

    Parameters
    ----------
    X : array-like (mxn)-shpae input
    y : array-like (mxk)-shape output
    theta : array-like (nxk)-shape input for weight
    lamb : float or int, optional
        Variable to calculate regularization. The default is 0.

    Returns
    -------
    J : float
        error between y_pred and actual y

    '''
    m,n = X.shape
    y_pred = np.zeros((m,1))                            #create an empty variable for y
     
    y_pred = sigmoid(X, theta)                          #predict y value
    reg = (lamb/(2*m))*np.sum(np.square(theta[1:]))     #calculate regularization

    #find error
    J = -(1/m)*(np.dot(y.T,np.log10(y_pred))+np.dot(
        (1-y).T,np.log10(1-y_pred)))
    J = J + reg
    
    return J