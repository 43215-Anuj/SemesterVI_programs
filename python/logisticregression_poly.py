# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:22:33 2019

@author: BALJEET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt    # more on this later

data = pd.read_csv('ex2data2.txt', header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]
data.head()
Xtest=X
mask = y == 1
passed = plt.scatter(X[mask][0].values, X[mask][1].values)
failed = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()

#def mapFeature(X1, X2):
#    degree = 6
#    out = np.ones(X.shape[0])[:,np.newaxis]
#    for i in range(1, degree+1):
#        for j in range(i+1):
#            out = np.hstack((out, np.multiply(np.power(X1, i-j),np.power(X2, j))[:,np.newaxis]))
#    return out
#X = mapFeature(X.iloc[:,0], X.iloc[:,1])


#polynomial fit I did not use the map feature as students know about the polynomial features
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=6)

X = polynomial_features.fit_transform(X)


def sigmoid(x):
  return 1/(1+np.exp(-x))


def lrCostFunction(theta_t, X_t, y_t):
    m = len(y_t)
    J = (-1/m) * (y_t.T @ np.log(sigmoid(X_t @ theta_t)) + (1 - y_t.T) @ np.log(1 - sigmoid(X_t @ theta_t)))
    #reg = (lambda_t/(2*m)) * (theta_t[1:].T @ theta_t[1:])
    #J = J + reg
    return J



def lrGradientDescent(theta, X, y):
    m = len(y)
    grad = np.zeros([m,1])
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    #grad[1:] = grad[1:] + (lambda_t / m) * theta[1:]
    return grad

(m, n) = X.shape
y = y[:, np.newaxis]
theta = np.zeros((n,1))

J = lrCostFunction(theta, X, y)
print(J)


output = opt.fmin_tnc(func = lrCostFunction, x0 = theta.flatten(), fprime = lrGradientDescent, \
                         args = (X, y.flatten()))
theta = output[0]
print(theta) # theta contains the optimized values
J = lrCostFunction(theta, X, y)
print(J)


pred = [sigmoid(np.dot(X, theta)) >= 0.5]
np.mean(pred == y.flatten()) * 100


u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))
def mapFeatureForPlotting(X1, X2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
    return out
for i in range(len(u)):
    for j in range(len(v)):
        z[i,j] = np.dot(mapFeatureForPlotting(u[i], v[j]), theta)
mask = y.flatten() == 1
X = data.iloc[:,:-1]
passed = plt.scatter(X[mask][0], X[mask][1])
failed = plt.scatter(X[~mask][0], X[~mask][1])
plt.contour(u,v,z,0)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()
