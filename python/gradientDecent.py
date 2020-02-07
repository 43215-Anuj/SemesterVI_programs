"""
Created on Tue Jan 21 13:15:50 2020
@author: anujk
"""

# Gradient Decent Univariate Linear regression 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\anujk\\Documents\\python\\restorent.csv")
x = data.iloc[:,0]
y = (2*x[:,0] + 2*x[:,1]).reshape(96,1)

y = data.iloc[:,1]
m = len(y)
x = x[:, np.newaxis]
y = y[:, np.newaxis]

theta = np.zeros([2,1])
iteration = 10500
alpha = 0.0129999
ones = np.ones([m,1])

x = np.column_stack((ones,x))

def compute_cost(x,y,theta):
    # f(h(theta)) = (X.theta - y)
    residual = np.dot(x,theta) - y
    res_sum = np.sum(np.power(residual,2))
    cost = (res_sum)/(2*m)
    return cost

cost = compute_cost(x,y,theta)

def gradient_desent(x,y,theta,maxiter,alpha):
    m=len(y)
    for _ in range (maxiter):
        g1 = np.dot(x,theta) - y
        g2 = np.dot(x.T,g1)
        theta = theta - (alpha/m)*g2
        return theta

theta = gradient_desent(x,y,theta,iteration,alpha)

new_cost = compute_cost(x,y,theta)

plt.scatter(x[:,1],y)
plt.plot(x[:,1],np.dot(x,theta),color='r')