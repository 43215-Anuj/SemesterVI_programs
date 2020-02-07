#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:30:58 2020

@author: baljeetkaur
"""
import numpy as np
from sklearn.model_selection import KFold # import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
X = np.array([[11, 111], [22, 222], [33,333], [44, 4444]]) # create an array
y = np.array([1, 2, 3, 4]) # Create another array
kf = KFold(n_splits=2) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf) 



for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train = X[train_index] 
    X_test=X[test_index]
    y_train = y[train_index]
    y_test=y[test_index]
    print([X_train])
    print([X_test])
    print([y_train])
    print([y_test])
    input()
    
    
    
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train = X[train_index] 
    X_test=X[test_index]
    y_train = y[train_index]
    y_test=y[test_index]
    model = LinearRegression()   #LOW TRAIN AND TESY
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)

    rmse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)

    print('Root mean squared error train: ', rmse)
    print('R2 score train: ', r2)

    y_test_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    print('Root mean squared error test: ', rmse)
    print('R2 score  test: ', r2)
    
    
    