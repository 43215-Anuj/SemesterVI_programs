# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:17:38 2020

@author: anujk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

X=(np.random.normal(0,1,100)).reshape(-1,1)
y=3*(X**3)+4*(X**2)+5*(X)+(np.random.normal(0,3,100)).reshape(-1,1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)

polynomial_features=PolynomialFeatures(degree=20)
x_poly=polynomial_features.fit_transform(X)
model=LinearRegression()

mse=r2=mse_test=r2_test=0
for _ in range(10):

    X_train,X_test,y_train,y_test=train_test_split(x_poly,y,train_size=0.8)

    model.fit(X_train,y_train)
    
    y_pred=model.predict(X_train)
    mse+=mean_squared_error(y_train,y_pred)
    r2+=r2_score(y_train,y_pred)
    
    y_pred_test=model.predict(X_test)
    mse_test+=mean_squared_error(y_test,y_pred_test)
    r2_test+=r2_score(y_test,y_pred_test)
    
mse/=10
mse_test/=10
r2/=10
r2_test/=10

plt.scatter(X,y,s=10)
z=np.hstack((X_train[:,1].reshape(-1,1),y_pred))
z=z[z[:,0].argsort()]
plt.plot(z[:,0],z[:,1],color='r')
