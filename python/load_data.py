#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:12:15 2020

@author: baljeetkaur
"""

# Load scikit-learn's datasets
from sklearn import datasets
# Load digits dataset
digits = datasets.load_digits()
# Create features matrix
features = digits.data
# Create target vector
target = digits.target
# View first observation
features[0]

#load_boston load_iris


#simulated for regression
# Load library
from sklearn.datasets import make_regression
# Generate features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(n_samples = 100,n_features = 3,n_informative = 3,n_targets = 1,noise = 0.0,coef = True,random_state = 1)
# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

#simulated for classification
# Load library
from sklearn.datasets import make_classification
# Generate features matrix and target vector
features, target = make_classification(n_samples = 100,n_features = 3,n_informative = 3,n_redundant = 0,n_classes = 2,weights = [.25, .75],random_state = 1)
# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])


#clustering
# Load library
from sklearn.datasets import make_blobs
# Generate feature matrix and target vector
features, target = make_blobs(n_samples = 100,n_features = 2,centers = 3,cluster_std = 0.5,shuffle = True,random_state = 1)
# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])


# Load library
import matplotlib.pyplot as plt
# View scatterplot
plt.scatter(features[:,0], features[:,1], c=target)
plt.scatter(features[:,0], features[:,1])
plt.show()

#load csv using pandas
# Load library
import pandas as pd
# Create URL
url = 'https://tinyurl.com/simulated_data'
# Load dataset
dataframe = pd.read_csv(url)
# View first two rows
dataframe.head(2)

