import matplotlib.pyplot as plt
import numpy as np
import math

def phi(x): return 20 * math.pi * x*x*x
def phiprime(x): return 20 *3* math.pi * x*x
def phiprimeprime(x): return 20 *6* math.pi * x

def u(x):  return 1 + 12*x - 10 *x*x + .5* math.sin(phi(x))
def f(x): return -20 + .5* phiprimeprime(x) *math.cos(phi(x)) - .5*(phiprime(x)*phiprime(x) ) *math.sin(phi(x))

def jacobi( F, uold, maxiter,h):
    for iter in np.arange(0,maxiter):
        unew=np.copy(uold)
        for i in np.arange(1, len(uold)-1):
            unew[i] =( 0.5*(uold[i-1] + uold[i+1]  - h*h  * F[i-1]))
        uold = np.copy(unew)
    return uold

m= 100
h= 1/(m+1)
v= np.linspace(0,1, m+2) # the inital u guess with the boundary points
x = v[1:(m+1)]

F = [f(i) for i in x]

ua = jacobi(F, v*2+1, 20001,h)

exact = [u(i) for i in x]

plt.figure()
plt.plot(x, exact, "black", lw=2, label='exact')
plt.plot(v,ua, label = "jacobi ")
plt.show()
