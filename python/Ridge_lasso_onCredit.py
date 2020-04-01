import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV

dt = pd.read_csv("D://Project Work//Study Material//SemesterVI_programs//python//Credit.csv")
dt.info()

X = dt.drop(["Gender","Student","Married","Ethnicity","Balance"], axis=1)
X = X.drop([X.columns[0]], axis=1)

Y = dt.Balance

alphas = 10**np.linspace(10,-10,100)*0.5

ridge = Ridge(normalize = True)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X,Y)
    coefs.append(ridge.coef_)

y=np.array([1,2,3,4,5,6,7,8,9,10])
ax = plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale("log")
plt.axis("tight")
plt.xlabel('alphas')
plt.ylabel('coefs')
plt.title('Ridge regression on Credit data')

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2)

#Applying Ridge on random alpha
ridge2 = Ridge(alpha = 0.005, normalize=True)
ridge2.fit(X_train,Y_train)
pred = ridge2.predict(X_test)
print(pd.Series(ridge2.coef_ , index = X.columns))
print(mean_squared_error(Y_test,pred))

#FINDING THE BEST ALPHA FOR THE MODEL USING CROSS VALIDATION 
ridgeCV = RidgeCV(alphas = alphas, scoring="neg_mean_squared_error", normalize = True)
ridgeCV.fit(X_train,Y_train)
ridge_best = Ridge(alpha = ridgeCV.alpha_, normalize=True)
ridge_best.fit(X_train,Y_train)
pred_best = ridge_best.predict(X_test)
print("best value of alpha :",ridgeCV.alpha_)
print("rmse corrosponding to this alpha ",mean_squared_error(Y_test,pred_best))
