# Evan Waldmann
# 3-21-19
# 4.1 Homework

#https://barbagroup.github.io/essential_skills_RRC/laplace/1/

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
    for i in arange(0, len(x)-1)
        newx.append(x[i])
        newx.append((x[i]+ x[i+1])/2)
    return newx

def step(U0, RHS, level,h, A):
    if level ==1: #base case
        return U0

    I =
    R =1/2 * np.transpose(I)
    #step 1
    uv = bookjac(RHS, U0m , 3, h)  #get approximation to next u
    #step 2
    rv = RHS - A*uv #compute residual
    #step 3
    crv = rv[::2]  #choose everyother rv to coarsen the grid
    cA = R*A*I # dont know if this will work
    #step 4
    # ce = sla.inverse(cA, -crv) # solve Ae = - r
    ce =  bookjac(u0, rhs, 3, h)
    #step 5
    e = interpolateMidpoints(ce)
    uv = uv + e
    #step 6
    #step 4
    step(uv, rhs, level-1, h)


m= 64
h= 1/(m)
x =np.arange(0,1+h , h)
x=np.copy(x[1:m])
X.append(x)
m=m-1

k = [1,-2,1]
offset = [-1,0,1]
A2 = (1/(h*h) * sp.diags(k,offset, (m,m)).todense())
F = [f(i) for i in x]

v= 3 # iterations
omega = 2/3

initialValue = zeros(m)

#A U = RHS with U0 intial value
step(initialValue , F , 6, h, A2 )

# Abar = R* A* I

#graph solution

#log log error plot
