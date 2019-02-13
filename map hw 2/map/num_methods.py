# Evan Waldmann
# 2-13-19
# 2.3 Homework

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import math

n = 100 #n = 1/h
epsarr = [1e-1,1e-2 ,1e-3, 1e-4]
eps = .001
print(eps)

# f and F vector
def f(x):return 0
rg= np.arange(0.01,.99 , .01)
F = [(f(0))]+ [f(i) for i in rg ] + [f(1)- 1/.01/.01]
print("F: ")
print(F)

U= []
count =0
for eps in epsarr:
    #make A tridiagonal matrix
    k = np.array([(2 - 1/eps /n)* np.ones(n-1) ,-4*np.ones(n),(2 + 1/eps /n)*np.ones(n-1)])
    offset = [-1,0,1]
    A = - 1* eps* n*n /2 * sp.diags(k,offset).toarray()
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
plt.plot(x, U[0], 'b', lw=2, label='U b e-1')
plt.plot(x, U[1], 'r', lw=2, label='U b e-2')
plt.plot(x, U[2], 'g', lw=2, label='U b e-3')
plt.plot(x, U[3], 'pink', lw=2, label='U b e-4')
plt.legend(loc='best')
plt.show()

print("Scheme from part b: ")

U1= []
count =0
for eps in epsarr:
    #make A tridiagonal matrix
    k = np.array([(2- 1/eps /n)* np.ones(n-1) ,(-4 + 1/n/eps)*np.ones(n),(2)*np.ones(n-1)])
    offset = [-1,0,1]
    A = -1* eps* n*n * sp.diags(k,offset).toarray()
    print("A:")
    print(A)

    print()

    # # f and F vector
    # def f(x):return 0
    # rg= np.arange(0.01,.99 , .01)
    # F = [(f(0))]+ [f(i) for i in rg ] + [f(1)- 1*n*n]
    # print("F: ")
    # print(F)

    #solve system to get U
    U1.append( spa.spsolve(A, F))
    print("U: ")
    print(U1[count])
    count = count + 1

#plotting
plt.figure()
plt.plot(x, U1[0], 'b', lw=2, label='U b e-1')
plt.plot(x, U1[1], 'r', lw=2, label='U b e-2')
plt.plot(x, U1[2], 'g', lw=2, label='U b e-3')
plt.plot(x, U1[3], 'pink', lw=2, label='U b e-4')
plt.legend(loc='best')
plt.show()
