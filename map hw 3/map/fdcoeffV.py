import numpy as np
import math
import scipy.linalg as sla

 # k=difference order , xbar =point for stencil, x = array of stencil points
def fdcoeffV(k, xbar, x):
    n = len(x)
    print(n)
    A = np.ones((n,n))
    xrow = np.transpose( x - xbar)
    for i in np.arange(1, n):
        A[i]= (np.power(xrow,(i))) / math.factorial(i)

    b= np.zeros(n)
    b[k] = 1 # kth derivative term is not zeros
    print(A)
    print(b)
    c = sla.solve(A, b)
    return(np.transpose(c))

n= 3
x=np.arange(1,n+1)
print(x)
coef = fdcoeffV(2, 0,x) # evaluates the 2 derivative central difference.
print(coef)
