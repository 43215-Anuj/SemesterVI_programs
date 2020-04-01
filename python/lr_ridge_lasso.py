#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:07:09 2020

@author: baljeetkaur
"""
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score


from sklearn.metrics import mean_squared_error, r2_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lr = linear_model.LinearRegression()


from sklearn.model_selection import ShuffleSplit
#cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cv = ShuffleSplit(n_splits=10, test_size=0.1)
cross_val_score(lr, X, y, cv=cv)
np.mean(cross_val_score(lr, X, y, cv=cv))


myridge = linear_model.Ridge(alpha=0.1)
cross_val_score(myridge, X, y, cv=cv)
np.mean(cross_val_score(myridge, X, y, cv=cv))
