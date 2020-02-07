# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:25:34 2020

@author: Archit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


data=pd.read_csv("C:\python\ML.csv")
X=(data.iloc[:,0])
print(X)
Y=data.iloc[:,1]
print(Y)
X=X[:,np.newaxis]
Y=Y[:,np.newaxis]

regression_model=LinearRegression()
regression_model.fit(X,Y)
Y_predict=regression_model.predict(X)
mse=mean_squared_error(Y,Y_predict)
r2=r2_score(Y,Y_predict)
 
plt.scatter(X,Y)
plt.plot(X,Y)

plt.scatter(X,Y_predict)
plt.plot(X,Y_predict)
