# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:02:27 2020

@author: anujk
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
datatrain = pd.read_csv('../Datasets/iris/iris.csv')

# Change string value to numeric
datatrain.loc[datatrain['species']=='Iris-setosa', 'species']=0
datatrain.loc[datatrain['species']=='Iris-versicolor', 'species']=1
datatrain.loc[datatrain['species']=='Iris-virginica', 'species']=2
datatrain = datatrain.apply(pd.to_numeric)

# Change dataframe to array
datatrain_array = datatrain.values

# Split x and y (feature and target)
X_train, X_test, y_train, y_test = train_test_split(datatrain_array[:,:4],
                                                    datatrain_array[:,4],
                                                    test_size=0.2)

"""
SECTION 2 : Build and Train Model
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 10 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris, Softmax Layer
optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.01
epoch = 500
"""

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10),
                    solver='sgd',
                    learning_rate_init=0.01,
                    max_iter=500,
                    random_state=113)

# Train the model
mlp.fit(X_train, y_train)

# Test the model
print mlp.score(X_test,y_test)

sl = 5.8
sw = 4
pl = 1.2
pw = 0.2
data = [[sl,sw,pl,pw]]
print mlp.predict(data)
