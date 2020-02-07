# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:33:07 2020

@author: Rashmi Mishra
"""

import numpy as np
data=np.random.rand(100,3)
X=np.column_stack((np.ones(100),data))
Y=(5*X[:,0]+7.2*X[:,1]+9.2*X[:,2]+7*X[:,3])+np.random.rand(1)
ans=np.linalg.inv(X.T@X)@X.T@Y
ans
