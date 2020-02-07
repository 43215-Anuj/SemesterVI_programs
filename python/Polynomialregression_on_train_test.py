import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

#X=2-3*np.random.normal(0,5,100).reshape(-1,1)
#Y= 5*X+2*(X**2)+0.5*(X**3)+(np.random.normal(0,3,100).reshape(-1,1))

X=(np.random.normal(0,1,100)).reshape(-1,1)
Y=3*(X**3)+4*(X**2)+5*(X)+(np.random.normal(0,3,100)).reshape(-1,1)
#divide into train and test
scaler = StandardScaler()
X = scaler.fit_transform(X)

polynomial_feature = PolynomialFeatures(degree=4)
X_poly = polynomial_feature.fit_transform(X)
model=LinearRegression()

rmse = []
R2 = []
rmse_test = []
R2_test = []

for i in range(10):
    #polynomial regression on train data 
    X_train,X_test,Y_train,Y_test=train_test_split(X_poly,Y,train_size=0.2)
    
    model.fit(X_train,Y_train)
    
    Y_train_predict = model.predict(X_train)

    rmse.append(mean_squared_error(Y_train,Y_train_predict))
    R2.append(r2_score(Y_train,Y_train_predict))
    """
    Z = np.hstack((X_train[:,1].reshape(-1,1),Y_train_predict))
    Z = Z[Z[:,0].argsort()]
    plt.scatter(X,Y,s=10)
    plt.plot(Z[:,0],Z[:,1],color='r')
    """  
    
    #test Data
    Y_test_predict=model.predict(X_test)

    rmse_test.append(mean_squared_error(Y_test,Y_test_predict))
    R2_test.append(r2_score(Y_test,Y_test_predict))
    """
    Z = np.hstack((X_test[:,1].reshape(-1,1),Y_test_predict))
    Z = Z[Z[:,0].argsort()]
    plt.scatter(X,Y,s=10)
    plt.plot(Z[:,0],Z[:,1],color='r')
    """
mean_rmse = sum(rmse)/10
mean_r2 = sum(R2)/10
mean_rmse_test = sum(rmse_test)/10
mean_R2_test = sum(R2_test)/10