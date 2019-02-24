# Evan Waldmann
# 2-24-19
# 3.1 Homework

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import math

np.set_printoptions(linewidth=132)
################################################################################
#       U0
################################################################################
print("u0")
def u0(x):return np.sin(math.pi * x)
def f0(x):return ( math.pi* math.pi * np.sin(math.pi * x) )

ms = np.arange(10, 201, 10)
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
    A = -1* n*n * sp.diags(k,offset).toarray()

    F = [f0(i) for i in x]

    #solve system to get U
    U= spa.spsolve(A, F)
    exact0 = [u0(i) for i in x ]
    P2.append(np.linalg.norm(U-exact0, np.inf))

    # fourth order
    k = np.array([-1 *np.ones(n-2) , 16*np.ones(n-1) , -30*np.ones(n) ,16* np.ones(n-1) , -1*np.ones(n-2)]) # fourth order central
    offset = [-2,-1,0,1,2]
    A = -1/12* n*n * sp.diags(k,offset).toarray()

    U= spa.spsolve(A, F)
    exact = [u0(i) for i in x ]
    P4.append(np.linalg.norm(U-exact, np.inf))

    count = count + 1

# plt.figure()
# plt.plot(ms, P2, 'b', lw=2, label='U2')
# plt.plot(ms, P4, 'r', lw=2, label='U4')
# plt.legend(loc='best')
# plt.show()

plt.figure()
plt.loglog(ms, P2, 'b', lw=2, label='fd2')
plt.loglog(ms, P4, 'r', lw=2, label='fd4')
plt.title("Error of U0")
plt.legend(loc='best')
plt.show()

################################################################################
#       U1
################################################################################
print("u1")
def u(x):
    if (x <= .05):
        return np.sin(math.pi * x)
    else:
        return 4*x*(1-x)

def f(x):
    if (x > .5):
        return 8
    else:
        return ( math.pi* math.pi * np.sin(math.pi * x) )

ms = np.arange(10, 201, 10)
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
    A = -1* n*n * sp.diags(k,offset).toarray()

    F = [f0(i) for i in x]

    #solve system to get U
    U= spa.spsolve(A, F)
    exact0 = [u0(i) for i in x ]
    P2.append(np.linalg.norm(U-exact0, np.inf))

    # fourth order
    k = np.array([-1 *np.ones(n-2) , 16*np.ones(n-1) , -30*np.ones(n) ,16* np.ones(n-1) , -1*np.ones(n-2)]) # fourth order central
    offset = [-2,-1,0,1,2]
    A = -1/12* n*n * sp.diags(k,offset).toarray()

    U= spa.spsolve(A, F)
    exact = [u0(i) for i in x ]
    P4.append(np.linalg.norm(U-exact, np.inf))

    count = count + 1

# plt.figure()
# plt.plot(ms, P2, 'b', lw=2, label='U2')
# plt.plot(ms, P4, 'r', lw=2, label='U4')
# plt.legend(loc='best')
# plt.show()

plt.figure()
plt.loglog(ms, P2, 'b', lw=2, label='fd2')
plt.loglog(ms, P4, 'r', lw=2, label='fd4')
plt.title("Error of U1")
plt.legend(loc='best')
plt.show()
