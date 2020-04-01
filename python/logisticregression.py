# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:28:11 2019

@author: BALJEET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    print(acc * 100)
    
    

#hypothesis
def sigmoid(x):
  return 1/(1+np.exp(-x))

#cost functiom

def costFunction(theta, X, y):
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid(X @ theta))) 
        + np.multiply((1-y), np.log(1 - sigmoid(X @ theta))))
    return J  


def costFunction_bk(theta, X, y):
    J = (-1/m) * ((y.T@ np.log(sigmoid(X @ theta)))+((1-y).T@np.log(1 - sigmoid(X @ theta))))
    
    return J  



def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))



data = pd.read_csv('D://Project Work//Study Material//SemesterVI_programs//python//ex2data1.txt', header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]
data.head()


mask = y == 1
#X_feature1=X[0][mask];   
adm = plt.scatter(X[mask][0].values, X[mask][1].values)
not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()


(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]
theta = np.zeros((n+1,1)) # intializing theta with all zeros
J = costFunction(theta, X, y)
print(J)

#This is a vectorized code giving same result
J = costFunction_bk(theta, X, y)
print(J)

grad = gradient(theta, X, y)
print(grad)

temp = opt.fmin_tnc(func = costFunction, 
                    x0 = theta.flatten(),fprime = gradient, 
                    args = (X, y.flatten()))
#the output of above function is a tuple whose first element #contains the optimized values of theta
theta_optimized = temp[0]
print(theta_optimized)


J = costFunction(theta_optimized[:,np.newaxis], X, y)
print(J)


plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0] 
          + np.dot(theta_optimized[1],plot_x))  
mask = y.flatten() == 1
adm = plt.scatter(X[mask][:,1], X[mask][:,2])
not_adm = plt.scatter(X[~mask][:,1], X[~mask][:,2])
decision_boun = plt.plot(plot_x, plot_y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()
accuracy(X, y.flatten(), theta_optimized, 0.5)





