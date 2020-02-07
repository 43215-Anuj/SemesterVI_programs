# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:01:31 2020

@author: anujk
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# declare empty Array for storing data
rmse = []
r2 = []
rmse_test = []
r2_test = []

#Generating random data (X array) input attribute
X = (np.random.normal(0,10,100)).reshape(-1,1)
#Generating output data (Y array)
Y = 4*(X**3) + 0.5*(X**2) + 2*(X) + (np.random.normal(0,3,100)).reshape(-1,1)

#Scaling X array(Data)
scaler = StandardScaler()
X = scaler.fit_transform(X)
model = LinearRegression()

for i in range(15):
    #split Complete data into test and train data
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
    
    #applying regression, converting X into an 'i' dimension matrix 
    polynomial_features = PolynomialFeatures(degree=i+1)
    X_poly = polynomial_features.fit_transform(X_train)
    model.fit(X_poly,Y_train)
    Y_pred = model.predict(X_poly)
    
    rmse.append(mean_squared_error(Y_train,Y_pred))
    r2.append(r2_score(Y_train,Y_pred))
    Z = np.hstack((X_train[:,0].reshape(-1,1),Y_pred))
    Z = Z[Z[:,0].argsort()]
    plt.scatter(X,Y,s=10)
    plt.plot(Z[:,0],Z[:,1],color='r')
    
    X_poly_test = polynomial_features.fit_transform(X_test)
    model.fit(X_poly_test,Y_test)
    Y_pred_test = model.predict(X_poly_test)
    rmse_test.append(mean_squared_error(Y_test,Y_pred_test))
    r2_test.append(r2_score(Y_test,Y_pred_test))
    Z = np.hstack((X_test[:,0].reshape(-1,1),Y_pred_test))
    Z = Z[Z[:,0].argsort()]
    plt.scatter(X,Y,s=10)
    plt.plot(Z[:,0],Z[:,1],color='r')