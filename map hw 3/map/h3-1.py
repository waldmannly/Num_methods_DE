# Evan Waldmann
# 2-24-19
# 3.1 Homework

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import math

np.set_printoptions(linewidth=132)

# set up functions
def u0(x):return np.sin(math.pi * x)
def f0(x):return ( math.pi* math.pi * np.sin(math.pi * x) )
def u1(x):
    if (x <= .05):
        return np.sin(math.pi * x)
    else:
        return 4*x*(1-x)
def f1(x):
    if (x > .5):
        return 8
    else:
        return ( math.pi* math.pi * np.sin(math.pi * x) )

ms = np.arange(10, 201, 10)
# P2/P4 will hold the max norms for 2md and 4th order respectively
P2= []
P4= []
count =0
for m in ms:
    n = m
    h= 1/n
    x =np.arange(0,1 , h)

    #make A tridiagonal matrix
    k = np.array([np.ones(n-1),-2*np.ones(n),np.ones(n-1)]) #2nd order central
    offset = [-1,0,1]
    a = -1* n*n * sp.diags(k,offset)
    I = np.eye(m)
    # A = sp.bmat([[a, I],[I,a]]).todense()

    A= sp.bmat([[a if i==j else np.eye(n) if abs(i-j)==1
                  else None for i in range(2)] for j in range(2)])

    F = [f0(i) for i in x]+ [f1(i) for i in x]

    #solve system to get U
    U2= spa.spsolve(A, F)
    exact0 = [u0(i) for i in x ] + [u1(i) for i in x]
    P2.append(np.linalg.norm(U2-exact0, np.inf))

    # fourth order
    k = np.array([-1 *np.ones(n-2) , 16*np.ones(n-1) , -30*np.ones(n) ,16* np.ones(n-1) , -1*np.ones(n-2)]) # fourth order central
    offset = [-2,-1,0,1,2]
    a = -1/12* n*n * sp.diags(k,offset)

    # A = sp.bmat([[a, I],[I,a]]).todense()
    A= sp.bmat([[a if i==j else np.eye(n) if abs(i-j)==1
                      else None for i in range(2)] for j in range(2)])

    U4= spa.spsolve(A, F)
    exact = [u0(i) for i in x ]+ [u1(i) for i in x]
    P4.append(np.linalg.norm(U4-exact, np.inf))

    count = count + 1

plt.figure()
plt.plot(np.arange(0,2 , h), U2, 'b', lw=2, label='U2')
plt.plot(np.arange(0,2 , h), U4, 'r', lw=2, label='U4')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.loglog(ms, P2, 'b', lw=2, label='fd2')
plt.loglog(ms, P4, 'r', lw=2, label='fd4')
plt.title("Errors")
plt.legend(loc='best')
plt.show()
