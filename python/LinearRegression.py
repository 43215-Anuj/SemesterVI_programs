# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:29:54 2020

@author: anujk
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

data = pd.read_csv("C:\\Users\\anujk\\Documents\\python\\ML.csv")
x = data.iloc[:,0]
y = data.iloc[:,1]

x = x[:, np.newaxis]
y = y[:, np.newaxis]

regression_model = LinearRegression()
regression_model.fit(x,y)
y_predicted = regression_model.predict(x)
rmse = mean_squared_error(y,y_predicted)
r2 = r2_score(y,y_predicted)

plt.scatter(x,y_predicted)
plt.plot(x,y_predicted)





