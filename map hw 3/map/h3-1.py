# Evan Waldmann
# 2-24-19
# 3.1 Homework

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
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
for m in ms:
    h= 1/(m)
    x =np.arange(0,1 , h)

    #make A tridiagonal matrix
    k = [1,-2,1]
    offset = [-1,0,1]
    A = -1/(h*h) * sp.diags(k,offset, (m,m)).todense()

    F = [f0(i) for i in x]

    #solve system to get U
    U2= sla.solve(A, F)
    exact = [u0(i) for i in x ]
    P2.append(np.linalg.norm( A @ np.abs(U2-exact), np.inf))

    # fourth order
    offset = [-2,-1,0,1,2]
    diagonals = [-1,16,-30,16,-1]
    A4 =  (-1/(12*h*h)) * (sp.diags(diagonals , offset,(m,m)).todense())

    U4= sla.solve(A4, F)
    P4.append(np.linalg.norm( A4 @ np.abs(U4-exact), np.inf))

plt.figure()
plt.plot(x, exact, "black", lw=3, label='exact')
plt.plot(x, U2, 'b', lw=2, label='U2')
plt.plot(x, U4, 'r', lw=2, label='U4')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.loglog(1/ms, P2, 'b', lw=2, label='fd2')
plt.loglog(1/ms, P4, 'r', lw=2, label='fd4')
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
P22= []
P44= []

for m in ms:
    h= 1/(m)
    x =np.arange(0,1 , h)

    #make A tridiagonal matrix
    k = [1,-2,1]
    offset = [-1,0,1]
    A2 = (-1/(h*h) * sp.diags(k,offset, (m,m)).todense())

    F = [f(i) for i in x]

    #solve system to get U
    U2= sla.solve(A2, F)
    exact = [u(i) for i in x ]
    P22.append(np.linalg.norm(A2 @ np.abs(U2-exact), np.inf))

    # fourth order
    offset = [-2,-1,0,1,2]
    diagonals = [-1,16,-30,16,-1]
    A4 = ( (-1/(12*h*h)) * (sp.diags(diagonals , offset,(m,m))).todense())

    U4= sla.solve(A4, F)
    P44.append(np.linalg.norm(A4 @ np.abs( U4-exact ), np.inf))

plt.figure()
plt.plot(x, exact, "black", lw=3, label='exact')
plt.plot(x, U2, 'b', lw=2, label='U2')
plt.plot(x, U4, 'r', lw=2, label='U4')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.loglog(1/ms, P22, 'b', lw=2, label='fd2')
plt.loglog(1/ms, P44, 'r', lw=2, label='fd4')
plt.title("Error of U1")
plt.legend(loc='best')
plt.show()
