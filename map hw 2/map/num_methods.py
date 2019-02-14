# Evan Waldmann
# 2-13-19
# 2.3 Homework

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import math

n = 100
epsarr = [.1,.01 ,.001, .0001]
h=1/(n)

U= []
count =0
for eps in epsarr:

    # f and F vector
    def f(x):return 0
    rg= np.arange(0.01,.99 , .01)
    F = [(f(0))]+ [f(i) for i in rg ] + [f(1)+ eps/h/h+1/2/h]
    print("F: ")
    print(F)

    #make A tridiagonal matrix
    k = np.array([(eps - h/2)* np.ones(n-1) ,-2*eps*np.ones(n),(eps + h/2)*np.ones(n-1)])
    offset = [-1,0,1]
    A =  - 1/h/h * sp.diags(k,offset).toarray()
    print("A:")
    print(A)

    print()

    #solve system to get U
    U.append( spa.spsolve(A, F))
    print("U: ")
    print(U[count])
    count = count + 1

#plotting
plt.figure()
x =np.arange(0,1 , .01)
plt.plot(x, U[3], 'pink', lw=2, label='U b e-4')
plt.plot(x, U[2], 'g', lw=2, label='U b e-3')
plt.plot(x, U[1], 'r', lw=2, label='U b e-2')
plt.plot(x, U[0], 'b', lw=2, label='U b e-1')
plt.legend(loc='best')
plt.show()

print("\n\n\n")
print("Scheme from part b: ")

U1= []
count =0
for eps in epsarr:
    # f and F vector
    print(eps)
    def f1(x):return 0
    rg= np.arange(0.01,.99 , .01)
    F1 = [(f1(0))]+ [f1(i) for i in rg ] + [f1(1)+ eps/(h*h) + 1/2/h]
    print("F1: ")
    print(F1)

    #make A tridiagonal matrix
    # k = np.array([(eps - h/2)* np.ones(n-1) ,-2*eps*np.ones(n),(eps + h/2)*np.ones(n-1)])
    k = np.array([  (eps - h)* np.ones(n-1) ,(-2*eps + h) *np.ones(n), (eps)*np.ones(n-1)])
    offset = [-1,0,1]
    A =  - 1/(h*h) * sp.diags(k,offset).toarray()
    print("A:")
    print(A)

    print()
    #solve system to get U
    U1.append( spa.spsolve(A, F1))
    print("U: ")
    print(U1[count])
    count = count + 1

#plotting
plt.figure()
plt.plot(x, U1[3], 'pink', lw=2, label='U b e-4')
plt.plot(x, U1[2], 'g', lw=2, label='U b e-3')
plt.plot(x, U1[1], 'r', lw=2, label='U b e-2')
plt.plot(x, U1[0], 'b', lw=2, label='U b e-1')
plt.legend(loc='best')
plt.show()
