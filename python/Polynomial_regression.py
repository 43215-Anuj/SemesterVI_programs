# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:37:43 2020

@author: HR LAB-3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

np.random.seed(0)
X=2-3*np.random.normal(0,1,20)
Y=X-2*(X**2)+0.5*(X**3)+np.random.normal(-3,3,20)
plt.scatter(X,Y,s=10)

X=X[:,np.newaxis]
Y=Y[:,np.newaxis]

regression_model=LinearRegression()
regression_model.fit(X,Y)
Y_predict=regression_model.predict(X)
mse=mean_squared_error(Y,Y_predict)
r2=r2_score(Y,Y_predict)

print(mse)
print(r2)

plt.scatter(X,Y,s=10)
plt.plot(X,Y_predict,color='r')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
polynomial_feature=PolynomialFeatures(degree=2)
X_poly=polynomial_feature.fit_transform(X)

model=LinearRegression()
model.fit(X_poly,Y)
Y_poly_predict=model.predict(X_poly)
mse=mean_squared_error(Y,Y_poly_predict)
r2=r2_score(Y,Y_poly_predict)

print(mse)
print(r2)


from sklearn.preprocessing import PolynomialFeatures
polynomial_feature=PolynomialFeatures(degree=3)
X_poly=polynomial_feature.fit_transform(X)

model=LinearRegression()
model.fit(X_poly,Y)
Y_poly_predict=model.predict(X_poly)
mse=mean_squared_error(Y,Y_poly_predict)
r2=r2_score(Y,Y_poly_predict)

print(mse)
print(r2)


from sklearn.preprocessing import PolynomialFeatures
polynomial_feature=PolynomialFeatures(degree=10)
X_poly=polynomial_feature.fit_transform(X)

model=LinearRegression()
model.fit(X_poly,Y)
Y_poly_predict=model.predict(X_poly)
mse=mean_squared_error(Y,Y_poly_predict)
r2=r2_score(Y,Y_poly_predict)

print(mse)
print(r2)


from sklearn.preprocessing import PolynomialFeatures
polynomial_feature=PolynomialFeatures(degree=20)
X_poly=polynomial_feature.fit_transform(X)

model=LinearRegression()
model.fit(X_poly,Y)
Y_poly_predict=model.predict(X_poly)
mse=mean_squared_error(Y,Y_poly_predict)
r2=r2_score(Y,Y_poly_predict)

print(mse)
print(r2)

from sklearn.preprocessing import PolynomialFeatures
polynomial_feature=PolynomialFeatures(degree=8)
X_poly=polynomial_feature.fit_transform(X)

model=LinearRegression()
model.fit(X_poly,Y)
Y_poly_predict=model.predict(X_poly)
mse=mean_squared_error(Y,Y_poly_predict)
r2=r2_score(Y,Y_poly_predict)

print(mse)
print(r2)
