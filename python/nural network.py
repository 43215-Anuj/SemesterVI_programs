# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:13:40 2020

@author: HR LAB-3
"""

import numpy as np

X = np.array([
                [1,0,0],
                [1,0,1],
                [1,1,0],
                [1,1,1]
            ])

Y = np.array([0,1,1,1])
Y_pred = np.array([0,0,0,0])

weights = np.random.rand(3)
for i in range(len(weights)):
    weights[i] = weights[i] - np.random.rand(1)

def predict(X, weights):
    output = weights[0]
    for i in range(len(weights)-1):
        output =  weights[0] * X[:,0] + weights[1] * X[:,1] + weights[2] * X[:,2]
    
    for i in range(len(output)):
        if output[i] >= 0:
            Y_pred[i] = 1
        else:
            Y_pred[i] = -1
            
    
    
predict(X,weights)

for i in range(len(weights)):
    weights = weights + 0.5*(Y[i] - Y_pred[i]) * X[i]

