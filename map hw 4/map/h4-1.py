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

ms = [25,50,100]

# 1 - second order finite difference
U=[]
X =[]
for m in ms:
    m= m
    h= 1/(m)
    x =np.arange(0,1+h , h)
    x=np.copy(x[1:m])
    X.append(x)
    m=m-1

    #make A tridiagonal matrix
    k = [1,-2,1]
    offset = [-1,0,1]
    A2 = (1/(h*h) * sp.diags(k,offset, (m,m)).todense())

    F = [f(i) for i in x]
    F[0] = f(x[0]) - 1/h/h
    F[-1] = f(x[-1])  -1/h/h*3
    #solve system to get U
    U.append( sla.solve(A2, F))


plt.figure()
exact = [u(i) for i in x]
plt.plot(x, exact, "black", lw=3, label='exact')
plt.plot(X[0], U[0], 'b', lw=2, label='approximate m=25')
plt.plot(X[1], U[1], 'r', lw=2, label='approximate m=50')
plt.plot(X[2], U[2], 'pink', lw=2, label='approximate m=100')
plt.suptitle('Second Order Finite Difference')
plt.legend(loc='best')
plt.show()





def bookjac( F, uold, maxiter, m, h, x, exact):
    norms = []
    uold1 = uold
    for iter in np.arange(0,maxiter):
        unew=[]
        unew.append(uold[0])
        for i in np.arange(1, len(uold)-1):
            unew.append( 0.25*(uold[i-1] + uold[i+1] + uold[i] + uold[i] - h*h  * F[i]))
        unew.append(uold[len(uold)-1])
        # print(unew)
        uold = np.copy(unew)
        if (iter% 100 == 0):
            norms.append(norm( exact - uold, 2))
        if (iter % 2000 == 0 ):
            plt.plot(x, uold, label=str(iter))
    return norms

U=[]
X =[]
count = 0
for m in ms:
    m= m
    h= 1/(m)
    x =np.arange(0,1+h , h)
    p =np.arange(0,1+h , h)
    x=np.copy(x[1:m])
    X.append(x)
    m=m-1

    F = [f(i) for i in x]
    F[0] = f(x[0]) - 1/h/h
    F[-1] = f(x[-1])  -1/h/h*3

    plt.subplot(1,2,1)
    exact = [u(i) for i in x]
    plt.plot(x, exact, "black", lw=3, label='exact')
    U.append( bookjac(F, x*2+1, 20001, m,h,x, exact))
    plt.legend(loc='upper left')
    plt.suptitle('Jacobi method m= {}'.format(m+1))

    plt.subplot(1,2,2)
    plt.plot(np.arange(0,20001, 100), U[count])
    plt.suptitle('Jacobi errors m= {}'.format(m+1))

    plt.show()
    count = count +1
