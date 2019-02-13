# I install some stuff in atomm f5 runs the current file
# there should be autocomplete (cntr+ alt +g ) as well and some tab things?
# atom also has cntr +] to indent many lines .

#python
#ipython
import numpy as np
import scipy.sparse
l = [1,2,3,4]
l

a = np.array([1,2,3,4])

a+a # adds


l+l # concatenates

l = [1,2,"hello", 4] # you can have different data type in lists

np.array([1,2,3,4])
a.dtype # to get data type of array

np.array([1,2,3,4.0])
a.dtype

np.array([1,2,3,4.0], dtype = np.float32)


a= scipy.sparse.diags([3,2],[-2,0], (4,4))

a.todense()

# there are solvers for dense and sparse linear systems.
#it just depense on what you want ot do

# there is difference between element wise multiplication and "Regular" matrix
# multiplication

print(np.zeros((4,4)))
print("\n")
print(np.eye(4))
print("\n")

a= np.arange(6).reshape((3,2))
print(a)
print("\n")
print(a.T)
print("\n")
print(a.sum())
print("\n")


def f(x):
    return x*x

print(f(3))

a = np.array([1,2,3])
print(f(a))

#print(f([1,2,3])) # cant do this on lists b/c multiplication on lists is not defined


# plotting

n=11
x = np.linspace(0,10,n)

#y= np.zeros(n)
#for i in range(n):
#    y(n) = np.cos(x[i])

#or
y= np.cos(x)
