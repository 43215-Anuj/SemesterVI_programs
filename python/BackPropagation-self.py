# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:54:49 2020
@author: anujk
"""
import numpy as np

dataset = np.array([[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]);

weight1 = np.random.randint(25,size=(2,3), dtype ='l')*0.01

weight2 = np.random.randint(25,size=(2,3), dtype ='l')*0.01

input1=dataset[:,0:2];

ones = np.ones((10,1))
input1= np.hstack((input1,ones))

input1_t=input1.T;
for i in range(10) :
    #Extracting each column one by one (Constructing Input Layer)
    X = input1_t[:,i]
    #multipling weights with inputs(Creating First Hidden Layer)
    val1=weight1.dot(input1_t[:,i])
    #Applying activation function(sigmoid)
    val1_transfer=np.array([transfer(xi) for xi in val1])    #BK
        
    #appending 1 for bias 
    input2=np.hstack((val1_transfer,1))
    #Sigma wixi for second row
    #Multipling weights with inputs(Creating Second Hidden Layer)
    val2=weight2.dot(input2)
    #Applying activation function(sigmoid)
    val2_transfer=np.array([transfer(xi) for xi in val2])    #BK
    
    #(val2_transfer)
    #Predicting the output by identifying the brightest node 
    predicted=val2_transfer.argmax()
    actual=dataset[i,2]
    print('Expected=%d, Got=%d' % (actual, predicted))

#Measure the error generated
def cost(predicted):
    error = predicted - input1_t
    
    
    
def transfer(activation):
	return 1.0 / (1.0 + np.exp(-activation))