# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:52:23 2020

@author: anujk
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
#X=2-3*np.random.normal(0,5,100).reshape(-1,1)
#Y= 5*X+2*(X**2)+0.5*(X**3)+(np.random.normal(0,3,100).reshape(-1,1))

X=(np.random.normal(0,1,100)).reshape(-1,1)
Y=3*(X**3)+4*(X**2)+5*(X)+(np.random.normal(0,3,100)).reshape(-1,1)
#divide into train and test
scaler = StandardScaler()
X = scaler.fit_transform(X)

polynomial_feature = PolynomialFeatures(degree=4)
X_poly = polynomial_feature.fit_transform(X)

lr = LinearRegression()
for i in range(2):
    #polynomial regression on train data 
    cv = ShuffleSplit(n_splits=10, test_size=0.3)
    r2_score = cross_val_score(lr, X_poly, Y, cv=cv)
    print(sum(r2_score)/10)

plt.scatter(X,Y,s=10)
