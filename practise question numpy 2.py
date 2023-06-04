#!/usr/bin/env python
# coding: utf-8

# numpy Arryas
# python obeject:
1.high level number objects integers,floating ,points
2.continers: list (contlessninsertion and append,) dictionaries(fast lookup)numpy  provides:
 
extension package to python for multi deminsional arrays
closer to hardware(efficiency)
designed fo scientific computation (convenience)
also known as arrays oriented computing
# In[5]:


import numpy as np
a = np.array([0, 1, 2,3])
print(a)

print(np.arange(10))


# why it is useful : memory - efficient container that provides fast numerical operations

# In[6]:


#python lists
L = range(1000)
get_ipython().run_line_magic('timeit', '[i**2 for i in L]')


# In[7]:


a = np.arange(1000)
get_ipython().run_line_magic('timeit', 'a**2')


# creating arrays

# 1.1 manual construction of arrays

# In[8]:


# 1-D
a = np.array([0, 1, 2, 3])
a


# In[9]:


#print deminsions

a.ndim


# In[10]:


#shape
a.shape


# In[11]:


len(a)


# In[12]:


# 2-D , 3-D.....

b = np.array([[0, 1, 2,], [3, 4, 5]])

b


# In[13]:


b.ndim


# In[14]:


b.shape


# In[15]:


len(b) # return the size of the first dimention
b


# In[16]:


c = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

c


# In[17]:


c.ndim


# In[18]:


c.shape


# In[19]:


len(c)


# # 1.2 Function for creating arrays

# In[20]:


# during arange function 

# arange is an array-valued varsion of the built-in python range function 

a = np.arange(10) # 0.....n-1
a


# In[21]:


b = np.arange(1, 10, 2) #start, end, (exclusive),step
b


# In[22]:


# using linespace 

a = np.linspace(0, 1, 6) #start, and, number of points
a


# In[23]:


# common arrays

a = np.ones((3, 3))
a


# In[24]:


b = np.zeros((3, 3))
b


# In[25]:


# create array using randdom

# create an array  of the given shape and populates it with random sample from rand
a = np.random.rand(4)
a


# # 2. Basic Datatypes
you may have noticed that, in some instances array elements are desplayed with a trailing dot(e.g.2.vs 2). this is due to a differnce in the datatype used.
# In[26]:


a = np.arange(10)

a.dtype


# In[27]:


# you can explicitly specify which data_types you wants:

a = np.arange(10, dtype='float64')
a


# In[28]:


n = np.array((True, True,False,True)) #boolean datatype
print(n.dtype)


# In[29]:


d = np.array([1+2j, 2+4j]) #complex datatype

print(d.dtype)


# In[30]:


# for multidemensional arrays, indexs are tuples of integers:

a = np.diag([1, 2, 3])

print(a[2, 2])


# In[31]:


#array-wise comparisions
a = np.array([1, 2, 3, 4])
b = np.array([5, 2, 2, 4])
c = np.array([1, 2, 3, 4])

np.array_equal(a, b)


# # other reductions 

# In[32]:


x = np.array([1, 3, 2])
x.min()


# In[33]:


x.max()


# In[34]:


x.argmin() # index of minimum elements


# # statistics

# In[35]:


x = np.array([1, 2, 3, 1])
y = np.array([[1,2,3], [5,6,1]])
x.mean()


# In[36]:


np.median(x)


# In[37]:


a = np.tile(np.arange(0, 40,10), (3,1))
print(a)
print("************")
a=a.T
print(a)


# # flatting

# In[38]:


a = np.array([[1, 2, 3], [4, 5, 6]])
a.ravel() #Return a contigous flattened array. A 1-D array. containing the flatting


# In[39]:


a.T #Transpose


# In[40]:


a.T.ravel()


# Reshaping

# The inverse operation to flatting:

# In[41]:


print(a.shape)
print(a)


# In[42]:


b = a.ravel()
print(b)


# In[87]:


b = b.reshape((2, 3))
b


# note and beware:reshape may also return  copy!:

# In[91]:


a = np.zeros((3, 2))
b = a.T.reshape(3*2)
b[0] = 50
a


# In[92]:


a = np.arange(4*3*2).reshape(4, 3, 2)
a.shape


# In[93]:


a


# In[94]:


a.size


# # Sorting Data

# In[95]:


#sorting along an axis:
a = np.array([[5, 4, 6], [2, 3, 2]])
b = np.sort(a, axis=1)
b


# In[96]:


#sorting with fancy indiexing
a = np.array([4, 3, 1, 2])
j = np.argsort(a)
j

np.concatenate()
The concatenate() function is a function from the NumPy package. This function essentially combines NumPy arrays together. This function is basically used for joining two or more arrays of the same shape along a specified axis. There are the following things which are essential to keep in mind:

# In[1]:


import numpy as np  
x=np.array([[1,2],[3,4]])  
y=np.array([[12,30]])  
z=np.concatenate((x,y))  
z 


# In[3]:


#numpy.concatenate()with axis = 0


# In[6]:


import numpy as np
x=np.array([[1,2],[3,4]])
y=np.array([[12,30]])
z=np.concatenate((x,y),axis=0)
z


# In[8]:


#numpy.concatenate()with axis = 1


# In[9]:


import numpy as np
x=np.array([[1,2],[6,9]])
y=np.array([[18,60]])
z=np.concatenate((x,y.T),axis=1)
z


# In[11]:


#np.append() with axis=0 


# In[12]:


import numpy as np
a=np.array([[10,20,39],[50,60,30],[80,90,110]])
b=np.array([[11,14,15],[4,67,88],[73,83,93]])
c=np.append(a,b)
c=np.append(a,b,axis=0)
c

# numpy.reshape()in python

The numpy.reshape() function is available in NumPy package. As the name suggests, reshape means 'changes in shape'. The numpy.reshape() function helps us to get a new shape to an array without changing its data.

Sometimes, we need to reshape the data from wide to long. So in this situation, we have to reshape the array using reshape() function.
# 1)like index ordering 2)equivalent to raveral than reshape 3)fortran like index ordering

# In[2]:


import numpy as np
x=np.arange(12)
y=np.reshape(x,(4,3))
y=np.reshape((x),(3,4))
y=np.reshape(x,(4,3),order='f')
x
y


# # numpy.sum() in Python
# The numpy.sum() function is available in the NumPy package of Python. This function is used to compute the sum of all elements, the sum of each row, and the sum of each column of a given array.
# 
# Essentially, this sum ups the elements of an array, takes the elements within a ndarray, and adds them together. It is also possible to add rows and column elements of an array. The output will be in the form of an array object.
# 
# 1
# import numpy as np

# In[14]:


import numpy as np
a = np.array([0.6,0.9])
b = np.sum(a)
b


# In[15]:


import numpy as np
a = np.array([0.9,0.5,0.7,8.1])
x = np.sum(a,dtype=np.int32)
x


# # numpy.zeros without dtype and order
#  this function return a ndarray. the output array is the array with specified shape,dtype, order, and contains zeros.
# 

# In[16]:


import numpy as np  
a=np.zeros(12)  
a  


# In[17]:


#Create a 4*2 integer and prints its attribute


# In[19]:


import numpy 

firstArray = numpy.empty([4,2], dtype = numpy.uint16) 
print("Printing Array")
print(firstArray)

print("Printing numpy array Attributes")
print("1> Array Shape is: ", firstArray.shape)
print("2>. Array dimensions are ", firstArray.ndim)
print("3>. Length of each element of array in bytes is ", firstArray.itemsize)


# In[20]:


#Create a 5*2 integer array from a range between 100 to 200 such that the difference each elements is 10 .


# In[21]:


import numpy

print("Creating 5X2 array using numpy.arange")
sampleArray = numpy.arange(100, 200, 10)
sampleArray = sampleArray.reshape(5,2)
print (sampleArray)


# # numpy.random() in Python
# The random is a module present in the NumPy library. This module contains the functions which are used for generating random numbers. This module contains some simple random data generation methods, some permutation and distribution functions, and random generator functions.

# In[23]:


#p.random.rand(d0,d1,...dn)


# In[24]:


import numpy as np  
a=np.random.rand(5,2)  
a  


# # Joining NumPy Arrays
# Joining means putting contents of two or more arrays in a single array.
# 
# In SQL we join tables based on a key, whereas in NumPy we join arrays by axes.
# 
# We pass a sequence of arrays that we want to join to the concatenate() function, along with the axis. If axis is not explicitly passed, it is taken as 0.

# In[26]:


import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2))

print(arr)


# In[27]:


#join two 2D arrays along rows(axis=1)


# In[28]:


import numpy as np

arr1 = np.array([[1, 2], [3, 4]])

arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=1)

print(arr)


# # Splitting NumPy Arrays
# Splitting is reverse operation of Joining.
# 
# Joining merges multiple arrays into one and Splitting breaks one array into multiple.
# 
# We use array_split() for splitting arrays, we pass it the array we want to split and the number of splits.

# In[31]:


#split the array in 3 parts


# In[32]:


import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr)


# In[33]:


#write a numpy program to create vector the ranging from 252 to 1000 and print all values expect the first and last


# In[34]:


a = np.arange(252,1000)
print(a[0:-1:5])
a


# In[35]:


#Return array of odd rows and even columns from below numpy array


# In[36]:


import numpy

sampleArray = numpy.array([[3 ,6, 9, 12], [15 ,18, 21, 24], 
[27 ,30, 33, 36], [39 ,42, 45, 48], [51 ,54, 57, 60]]) 
print("Printing Input Array")
print(sampleArray)

print("\n Printing array of odd rows and even columns")
newArray = sampleArray[::2, 1::2]
print(newArray)


# write a numpy arrays

# 
