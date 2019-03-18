# Evan Waldmann
# 3-17-19
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
    A2 = (-1/(h*h) * sp.diags(k,offset, (m,m)).todense())

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



# 2 - jacobi
def jacobi(A, b, x0, tol, maxiter=200):
    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    k = 0
    rel_diff = tol * 2
    while (rel_diff > tol) and (k < maxiter):
        for i in range(0, n):
            subs = 0.0
            for j in range(0, n):
                if i != j:
                    subs += A[i,j] * x_prev[j]
            x[i] = (b[i] - subs ) / A[i,i]
        k += 1
        rel_diff = norm(x - x_prev) / norm(x)
        if k%2000 == 0:
            print(x, rel_diff)
        x_prev = x.copy()
    return x, rel_diff, k

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
    A2 = (-1/(h*h) * sp.diags(k,offset, (m,m)).todense())

    F = [f(i) for i in x]
    F[0] = f(x[0]) - 1/h/h
    F[-1] = f(x[-1])  -1/h/h*3
    #solve system to get U
    U.append( jacobi(A2, F, x*2+1,.000001, 2000))
print("here")
print((U[2][0]))

plt.figure()
exact = [u(i) for i in x]
plt.plot(x, exact, "black", lw=3, label='exact')
plt.plot(X[0], U[0][0], 'b', lw=2, label='approximate m=25')
plt.plot(X[1], U[1][0], 'r', lw=2, label='approximate m=50')
plt.plot(X[2], U[2][0], 'pink', lw=2, label='approximate m=100')
plt.suptitle('Jacobi method')
plt.legend(loc='best')
plt.show()

    # exact = [u(i) for i in x ]
    # P22.append(np.linalg.norm(A2 @ np.abs(U2-exact), np.inf))

    # I = np.arange(0, (m))
    # # J = np.arange(, (m+1))
    # U=np.zeros(m+1)
    # newU= np.zeros(m+1)
    # for iter in np.arange(0,maxiter):
    #     for i in I:
    #         newU[i] =  .5*(U[i-1] + U[i+1] - h*h* f(i))
    #     U = newU

# 3 - errors
