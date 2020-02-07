#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:00:35 2020

@author: baljeetkaur
"""

from sklearn.datasets import load_diabetes
 from sklearn.linear_model import RidgeCV
 X, y = load_diabetes(return_X_y=True)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)