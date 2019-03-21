# Evan Waldmann
# 3-21-19
# 4.1 Homework

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import math
from numpy.linalg import *

def phi(x): return 20 * math.pi * x*x*x
def phiprime(x): return 20 *3* math.pi * x*x
def phiprimeprime(x): return 20 *6* math.pi * x

def u(x):  return 1 + 12*x - 10 *x*x + .5* math.sin(phi(x))
def f(x): return -20 + .5* phiprimeprime(x) *math.cos(phi(x)) - .5*(phiprime(x)*phiprime(x) ) *math.sin(phi(x))

def bookjac( F, uold, maxiter, h):
    for iter in np.arange(0,maxiter):
        unew=[]
        unew.append(uold[0])
        for i in np.arange(1, len(uold)-1):
            unew.append( 0.25*(uold[i-1] + uold[i+1] + uold[i] + uold[i] - h*h  * F[i]))
        unew.append(uold[len(uold)-1])
        uold = np.copy(unew)
    return uold

def interpolateMidpoints(x):
    newx =[]
    for i in np.arange(0, len(x)-1):
        newx.append(x[i])
        newx.append((x[i]+ x[i+1])/2)
    return np.array(newx)

def step(U0, RHS, level,h, A):
    if level ==1: #base case
        return U0 # not really sure what this is returning or why it is a concatination of a bunch of vectors (this is were you are supposed to solve a small system, but i could not figure that part out)
    else:
        k = [1,2,1]
        offset = [-1,0,1]
        I = (1/2) * sp.diags(k,offset, (m,32)).todense()
        R =1/2 * np.transpose(I)
        #step 1
        uv = np.array(bookjac(RHS, U0 , 3, h))  #get approximation to next u
        #step 2
        rv = RHS - uv*A #compute residual
        #step 3
        crv = rv[::2]  #choose every other rv to coarsen the grid
        cA = R*A*I # coarsen A grid
        #step 4
        ce =  bookjac(cA, crv, 3, h)
        #step 5
        e = np.array(interpolateMidpoints(ce))[0] # could not get the I to work for me
        print(e)
        uv = uv + np.array(e)
        #step 6 from class book
        #step 4 from the slang notes
        #return uv[0]
        return step(uv, RHS, level-1, h,A) # i know you are suposed to call the step function in step 4 but i couldn't figure that out


m= 64
h= 1/(m)
x =np.arange(0,1+h , h)
x=np.copy(x[1:m])
m=m-1

k = [1,-2,1]
offset = [-1,0,1]
A2 = (1/(h*h) * sp.diags(k,offset, (m,m)).todense())
F = [f(i) for i in x]
v= 3 # iterations
omega = 2/3 # i dont know where you are supposed to use this value
initialValue = np.zeros(m)

#A U = RHS with U0 intial value
val = step(initialValue , np.array(F) , 6, h, A2 )
exact = sla.solve(A2,F)
#graph solution

plt.subplot(1,2,1)
plt.plot(x, exact, label="exact")
plt.plot(x, val[0])
plt.title("u value plot")

#log log error plot

plt.subplot(1,2,2)
plt.loglog(np.arange(0,63), exact)
plt.title("error plot")

plt.show()
