# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:12:04 2020

@author: Rashmi Mishra
"""

import numpy as np
from math import exp
from random import seed
from random import random

#sigmoid function as activation function
def sigmoid(x):
    return 1.0/(1.0+exp(-x))


#printing result after updated weights 
def print_result(weights):
         
       #creating prediction array to store predicted value
       results = list()
       print("Result for this iteration is ")
       #for every row in our dataset
       for row in data:
                   result = predict(weights, row); 
                   results.append(result); 
                   #displaying last row of dataset i.e actual and prediction value one by one
                   print('Result=%d, Output=%d' % (row[-1], result))
                   

#found it on google to return maximum from array you can calculate max and then return
def predict(weights, row):
    #Here we are getting outputs like we are gtting 2 ouputs for prediction we need maximum of that
    #we require to retun the indices of array with maximum 
	outputs = forward_propagation(weights, row);    print(max(outputs))
    #the way to get maximum get maximum from both indices of output    
	return outputs.index(max(outputs)) 
                   

#ativation fucntion whiuch multiplies input and weight
def function_activation(layer_weight, layer_input):
    #as we got two layer of weight
	a = layer_weight[-1]
    #now we are multiplying weights with our input i.e dot product of w and x
	for i in range(len(layer_weight)-1):
        #calculating activation and returning it
		a += layer_weight[i] * layer_input[i]
	return a

# Initialize a network list
def initialize_network(data,n_inputs, n_hidden, n_outputs):
	network = list()
     #here we require weights for hidden layer like no of input 2,
    #then add 1 whiuch is 3 running loop for that and then how many times we need is no of hidden layer i.e 2 
    # so total is 2*3=6 
	hidden_layer = [{'w':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
     #here we require weights for output layer like no of input 2,
    #then add 1 whiuch is 3 running loop for that and then how many times we need is no of output layer i.e 2 
    # so total is 2*3=6 
	output_layer = [{'w':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Initialize a network using array
def initialize_network1(data,n_inputs, n_hidden, n_outputs):
    #here we are initializing both the hidden layer nad input layer with weights
	neural = []   
	hidden_layer = np.array({'w':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden))
	neural.append(hidden_layer)   
	output_layer = np.array({'w':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs))
	neural.append(output_layer)
	return neural


# Initialize a network
def initialize_network2(data,n_inputs, n_hidden, n_outputs):
      n_hidden=2; weights = list()
      for i in range(n_inputs):
         hidden_layer=[{'w': np.random.rand(len(data[0]))}, 
                       {'w': np.random.rand(len(data[0]))}]; weights.append(hidden_layer)
      for i in range(n_hidden):
         output_layer=[{'w': np.random.rand(len(data[0]))}, 
                       {'w': np.random.rand(len(data[0]))}]; weights.append(output_layer)
         
         return weights
 

# Forward propagate input to a network output
def forward_propagation(data, row):
    #checking row by row
	answer = row
	for mind in data:
		new_answer = []
		for axion in mind:
            #value of activation
			activation = function_activation(axion['w'], answer)
            #output we get using sigmoid 
			axion['result'] = sigmoid(activation)
            #new input after hidden layer
			new_answer.append(axion['result'])
        #input to next layer    
		answer = new_answer
	return answer


# Calculate the derivative of an neuron output
def derivative(result):
	return result * (1.0 - result)


# Backpropagate error and store in neurons
def back_propagation(weights,actual):
    #we need to calculate error one by one in reverse
	for i in reversed(range(len(weights))):
        #starting from last one
		axion = weights[i]
        #generating error for one layer
		errors = list()
		for j in range(len(axion)):
				perceptron =axion[j]
                #calculating error by appending it by actual and reslut error
				errors.append(actual[j] - perceptron['result'])
            
		for j in range(len(axion)):
			perceptron = axion[j]
            #here getting error by multiply error and derivative 
			perceptron['error'] = errors[j] * derivative(perceptron['result'])
            

# Update network weights with error
def update_weights(data, row, l_rate):
	for i in range(len(data)):
		inputs = row[:-1]
		if i != 0:
			inputs = [mind['result'] for mind in data[i - 1]]
		for mind in data[i]:
			for j in range(len(inputs)):
                #update weights weights*error*inputs
				mind['w'][j] += l_rate * mind['error'] * inputs[j]
             #this one is for the row with values ones   
			mind['w'][-1] += l_rate * mind['error']
     
          
# Find a model
def neural_network(weights, data, learning_rate, n, n_outputs):
	for x in range(n):
        #suming error 
		sum_error = 0
		for row in data:
            #output from forward propagation
			outputs = forward_propagation(weights, row)
			actual = [0 for i in range(n_outputs)]
			actual[row[-1]] = 1; back_propagation(weights, actual)
            #total error (o-d)**2
			sum_error += 0.5* sum([(actual[i]-outputs[i])**2 for i in range(len(actual))])
            #updating new weights
			update_weights(weights, row, learning_rate); 
		print("\n",'n=%d, lrate=%.3f, error=%.3f' % (x, learning_rate, sum_error));print_result(weights)
         #displaying weights and expexcted output with actual

#dataset to find network
seed(1)
data = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

#inputs will be length of data-1
inputs = len(data[0]) - 1
outputs = len(data[0]) - 1

#initialize weights for data
weights = initialize_network(data,inputs, 2, outputs)

#neural network with 30 iteration and learning rate 0.5
neural_network(weights, data, 0.5, 30,outputs)

    
    
    
    
