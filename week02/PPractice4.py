# Info:

# Author:
# Date

# Purpose:

# inputs:

# outputs:

# Version control:

import numpy as np

#------------------------------------------------------------------------------
# Mean, Median and mode

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.3, 2.1, 30, 4.1, 6.6, 13])

a_mean = np.mean (y)
print ("the mean is:", a_mean)

a_median = np.median (y)
print ("the median is:", a_median)

# import lib
from scipy import stats

a_mode = stats.mode (y)
print ("the mode is:", a_mode)

# ## EXERCISE ## ================
    # create two 4*4 identity matrix
    # creat a 6*6 2darray (random) and a 2*2 array (eye), assign the 2by2 to 
    #       the middle of 6by6

m1 = np.random.random((6,6))  # Create an array filled with random values
print(m1) 

m2 = np.eye(2,2)  # Create an array filled with random values
print(m2) 

m1[2:4, 2:4] = m2
print(m1) 

print(m1[:,0]) 

# ## EXERCISE ## ================
    # find the mean, median and mode of whole array and different columns and rows


#------------------------------------------------------------------------------
# Mean, Mode
test_array = np.array([[1, 3, 4, 2, 2, 7],
                       [5, 2, 2, 1, 4, 1],
                       [3, 3, 2, 2, 1, 1]])

    
test_mean = np.mean(test_array, axis = 0) #axis = 0 columns
print (test_mean)

test_mean = np.mean(test_array, axis = 1) #axis = 1 rows
print (test_mean)

test_mean = np.mean(test_array[0,:])

test_median = np.median(test_array[0,:]) # along row 1
print (test_median)

a_mode = stats.mode (test_array[:,1])
print (a_mode)


#------------------------------------------------------------------------------
# Variance and STD

# creat a 2D array in the range of 10 to 20
m1 = np.random.random((4,3))*(20-10)+10 # Create an array filled with random values
print(m1) 

print(m1.shape) 

# what is the variance along first row
var_row1 = np.var (m1[0,:])
print (var_row1)

# what is the variance along first column
var_col1 = np.var(m1[:,0])
print(var_col1) 

# along all rows or columns
var1 = np.var(m1, axis=0) # columns
print (var1)

var2 = np.var(m1, axis=1) # rows
print (var2)

#------------------------------------------------------------------------------
# If conditions
x1 = 1
x2 = 2.5
if x1 > 10:
  print("value is greater than 10")
  
if x1 > 10:
    print("value is greater than 10")
else:
    print("value is not greater than 10")

if x1 > x2:
    print("value1 is greater than value2")
elif x1 == x2:
    print("two values are equal")
elif x2 < x1:
    print("value2 is greater than value1")


#------------------------------------------------------------------------------
# Loops
for i in (2,5):
    print(i)

for i in range(1,5):
    print (i)


# ask for 2 numbers from input
a = input () 
b = input () 

# creat the array
m1 = np.random.random((int(a),int(b)))*(20-10)+10 # Create an array filled with random values
print(m1) 

# go over the members of the array
for i in range(len(m1[0,:])):
    print(i)
    print(m1[0,i])














