# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:37:42 2020

@author: anujk
"""
# Comparing Gradient Decent with normal equation 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\anujk\\Documents\\python\\restorent.csv")
x = data.iloc[:,0]
ones = np.ones([96,1])
x = np.column_stack((ones,x))

y = (0.0743167*x[:,0] + 0.843559*x[:,1]).reshape(96,1)+ np.random.randn(96,1)
theta = np.zeros([2,1])

def normal(x,y,theta):
    x_trans = x.T
    x_trans_x= x_trans@x
    x_trans_x_inverse = np.linalg.inv(x_trans_x)
    x_trans_y = x_trans@y
    theta = x_trans_x_inverse@x_trans_y
    return theta

theta = normal(x,y,theta)

plt.scatter(x[:,1],y)
plt.plot(x[:,1],np.dot(x,theta),color='r')