# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:29:08 2019

@author: BALJEET
"""
# Load library
import numpy as np
#row vector
vector = np.array([1, 2, 3, 4, 5, 6])

vector1 = np.array([11,
                   22,
                   33,
                   44,
                   55,
                   66])
#colvector
colvector=np.array([[1],
                    [2],
                    [3],
                    [4]])

colvector1=np.array([[1], [2],  [3],  [4]])
#matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
#matrix object
matrixo = np.mat([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])


#sparse
from scipy import sparse
spmatrix = np.array([[0, 0, 3],
                   [0, 0, 0],
                   [0, 1, 0]])

sp_matr=sparse.csr_matrix(spmatrix)
print(sp_matr)

#select
# Create row vector
vector = np.array([1, 2, 3, 4, 5, 6])
vector[1]
matrix = np.array([[1.5, 2.6, 3.8],
                   [4, 5, 6],
                   [7, 8, 9]])
matrix[1,1]


matrix.shape
matrix.size
matrix.ndim



#Use NumPy’s vectorize:
# Load library
import numpy as np
# Create matrix
matrix = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
# Create function that adds 100 to something
add_100 = lambda i: i + 100
add_100(10)
# Create vectorized function
vectorized_add_100 = np.vectorize(add_100)
# Apply function to all elements in matrix
vectorized_add_100(matrix)


matrix + 100

#partial select

vector[:]
vector[:3]
vector[3:]
vector[-1]

#1:2 means leaving the second index, i think
matrix[:2,:]
matrix[:,1:2]


#min max
np.max(matrix)
np.min(matrix)


# Find maximum element in each column
np.max(matrix, axis=0)

# Find maximum element in each row
np.max(matrix, axis=1)


np.max(matrix,axis=0)
np.min(matrix,axis=0)


# Return mean
a=np.mean(matrix,axis=0)

# Return variance
np.var(matrix)


# Return standard deviation
np.std(matrix)

np.std(matrix,axis=0)


matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# Reshape matrix into 2x6 matrix
matrix.reshape(2, 6)

# Reshape matrix into 1 row and as many columns
matrix.reshape(3, -1)

#transpose
vector.T
np.array([[1,2,3,4,5]]).T   # note the difference

colvector.T
matrix.T
#flattening an array
matrix.flatten()
#or
matrix.reshape(1, -1)

#rank
np.linalg.matrix_rank(matrix)


# Create matrix
matrix = np.array([[1, 1, 1],
[1, 1, 10],
[1, 1, 15]])
# Return matrix rank
    
    matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])
np.linalg.matrix_rank(matrix)

#determnant
np.linalg.det(matrix)


#invserse
np.linalg.inv(matrix)

matrix@np.linalg.inv(matrix)

#diagonal
matrix.diagonal()

sum(matrix.diagonal())
#trace
matrix.trace()




#dotproduct sum aibi

vector_a = np.array([1,2,3])
vector_b = np.array([4,5,6])

np.dot(vector_a, vector_b)
vector_a @ vector_b



#adding matrix

# Create matrix
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])

# Create matrix
matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])

#add subtract multiply
np.add(matrix_a, matrix_b)
np.subtract(matrix_a, matrix_b) 


matrix_a+matrix_b
matrix_a-matrix_b


np.dot(matrix_a,matrix_b)
matrix_a@matrix_b


#element by element multiply
matrix_a*matrix_b


matrix@np.linalg.inv(matrix)

m=[[1,2],
   [2,5]]
m@np.linalg.inv(m)

#generating random values

np.random.seed(0)
#3 random integers between 0 and 4 
np.random.randint(0,5,3)


# 3 rnadom number normal dist mean 0 and stddev 1
np.random.normal(0.0,1.0,3)

# 3 rnadom number logistic dist mean 0 and scale 1
np.random.logistic(0.0,1.0,3)

# 3 rnadom number ge 1 lt 2
np.random.uniform(1.0,2.0,3)

