# Evan Waldmann
# 1-31-19
# 1.6 Homework

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import math

n = 100


#make A tridiagonal matrix
k = np.array([np.ones(n-1),-2*np.ones(n),np.ones(n-1)])
offset = [-1,0,1]
A = -1* n*n * sp.diags(k,offset).toarray()
print("A:")
print(A)

print()

# f and F vector
def f(x):return 1
rg= np.arange(0.01,.99 , .01)
F = [(f(0))]+ [f(i) for i in rg ] + [f(1)]
print("F: ")
print(F)

#solve system to get U
U= spa.spsolve(A, F)
print("U: ")
print(U)

#plotting
plt.figure()
x =np.arange(0,1 , .01)
plt.plot(x, U, 'b', lw=2, label='U1')
plt.legend(loc='best')
plt.show()

print("")
# new f and F vector
def f(x):return math.copysign(1, x-.5)
rg= np.arange(0.01,.99 , .01)
F = [(f(0))]+ [f(i) for i in rg ] + [f(1)]
print("F: ")
print(F)

#solve system to get U
U= spa.spsolve(A, F)
print("U: ")
print(U)

#plotting
plt.figure()
plt.plot(x, U, 'b', lw=2, label='U2')
plt.legend(loc='best')
plt.show()
