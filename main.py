# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 13:13:26 2022

@author: Xero
"""

#load required libraries
import pandas as pd                 #dataframe processing
import numpy as np                  #mathematical calculation
import matplotlib.pyplot as plt     #for data visualization
from sklearn.model_selection import train_test_split
from gradient import gradient
from sigmoid import sigmoid
from sklearn.metrics import accuracy_score
from predict import predict

#load dataset
df = pd.read_csv(r'C:\Users\Xero\Desktop\Ai\Dataset\age_interest_success.csv')
print(df.head())

#data visualization
plt.scatter(df['interest'],df['success'], marker='x')
plt.xlabel('Age')
plt.ylabel('Interest')
plt.title('Data Visualization')
plt.show()

#EDA analysis
print('\nExploraty Data Analysis')
print(df.skew())
print('\n---------------------')
print(df.corr())
print()

#x, y
X = df.drop(columns=['success']).to_numpy()
X = np.matrix(X)
y = df['success'].to_numpy()
y = np.matrix(y).T

m,n = X.shape

X0 = np.ones((m,1))              #create coeficient of intercept
X = np.c_[X0,X]                  #add 1s' column to 1st column of X
init_theta = np.zeros((n+1,1))   #initialize theta as zeros

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, 
                                                    random_state=42)

#size
m,n = X_train.shape

J_hist, theta = gradient(X_train,y_train,init_theta,lamb=0,
                         alpha=0.0005, max_iter=100)

print('---Trained Theta---')
print(theta)
print()

print('---Error---')
print(J_hist[-1])
print()

#learning Curve
plt.plot(np.arange(len(J_hist)),J_hist, lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Learning Curve')

#predict y value
y_pred = predict(X_train,theta)
print()
print("Accuracy score on train: ",accuracy_score(y_train,y_pred))

#predict y value
y_pred = predict(X_test, theta)
print("Accuracy score on test: ",accuracy_score(y_test,y_pred))