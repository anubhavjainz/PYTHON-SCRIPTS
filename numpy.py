# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:05:46 2019

@author: 555224
"""

import numpy as np


########    1 D ARRAY
np.arange(15)

#######    2 D ARRAY
a=np.arange(15).reshape(3,5)

a.shape

a.ndim

type(a)

a.itemsize

a.size

a.dtype.name

########### ARRAY CREATION


B=np.array([2,4,5])

B.dtype.name

b = np.array([1.2, 3.5, 5.1])

############ ERROR
a = np.array(1,2,3,4)

############ RIGHT
a = np.array([1,2,3,4])

b = np.array([[1.5,2,3], [4,5,6]])

c = np.array( [ [1,2], [3,4] ], dtype=complex )

c

np.zeros(10).reshape(2,5)

np.ones( (2,3,4), dtype=np.int16 )

np.empty( (2,3) ) 

np.empty( (2,3,4) ) 


np.arange(2,5)

np.arange(2,5,2)

############# POWER OPERATOR
b**2

############# CONDITIONAL OPERATOE

b>2


############# matrix operation

A = np.array( [[1,1],[0,1]] )

B= np.array([[2,0],[3,4]])

#elementwize product

A*B

#matrix product
A@B

#dot product

A.dot(B)

np.random.random((2,3))


########## Indexing, Slicing and Iterating

a = np.arange(10, dtype=np.int64)**3

a

a[2]

a[2:5]

# equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000
a[:6:2]

for i in a:
    print(i**(1/3))
    
###### floor function

np.floor([1.4,2.5,-3.4])   



###### random
mu, sigma = 2, 0.5
v = np.random.normal(mu,sigma,10000)
v 

a.transpose()