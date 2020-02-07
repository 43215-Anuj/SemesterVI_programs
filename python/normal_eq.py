# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:48:29 2020

@author: anujk
"""

import numpy as np

x = np.column_stack((np.ones()))
m = len(x)
ones = np.ones([m,1])
x = np.column_stack((ones,x))

y = (2*x[:,0] + 2*x[:,1]).reshape(5,1)

x_trans = x.T
x_trans_x= x_trans@x
x_trans_x_inverse = np.linalg.inv(x_trans_x)
x_trans_y = x_trans@y
theta = x_trans_x_inverse@x_trans_y
