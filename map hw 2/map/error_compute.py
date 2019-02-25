# Evan Waldmann
# 2-13-19
# 2.5 Homework

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import math

n = 100
epsarr = [.1,.01 ,.001, .0001]
h=1/(n)

U= []
Ainv =[]
Anorm =[]
tau =[]
count =0

print("part a")

for eps in epsarr:

    # f and F vector
    def f(x):return 0
    rg= np.arange(0.01,.99 , .01)
    F = [(f(0))]+ [f(i) for i in rg ] +[f(1)+ eps/h/h+1/2/h]

    #make A tridiagonal matrix
    k = np.array([(eps - h/2)* np.ones(n-1) ,-2*eps*np.ones(n),(eps + h/2)*np.ones(n-1)])
    offset = [-1,0,1]
    A =  - 1/h/h * sp.diags(k,offset).toarray()

    # calculate the analytic solution
    Uhat = (1- np.exp(-np.arange(0,1 , .01)/eps)) / (1- np.exp(-1/eps) )
    U= spa.spsolve(A,F)
    #compute truncation error
    tau.append( A @ (U- Uhat))
    print()
    print(eps)
    print("truncation error")
    print(sl.norm(tau[count], np.inf))

    count = count + 1


print("\n\n\n")
print("Scheme from part b: ")

U1= []
A1inv =[]
A1norm =[]
tau1 = []
count =0

print("part b")

for eps in epsarr:
    # f and F vector
    def f1(x):return 0
    rg= np.arange(0.01,.99 , .01)
    F1 = [(f1(0))]+ [f1(i) for i in rg ] + [f1(1)+ eps/(h*h) + 1/h]

    #make A tridiagonal matrix
    k = np.array([ (eps) * np.ones(n-1) ,(-2*eps - h) *np.ones(n),  (eps + h)*np.ones(n-1)])
    offset = [-1,0,1]
    A =  - 1/(h*h) * sp.diags(k,offset).toarray()

    U= spa.spsolve(A,F)
    # calculate the analytic solution
    Uhat = (1- np.exp( (-np.arange(0,1 , .01) / eps))) / (1- np.exp(-1/eps) )
    #compute truncation error
    tau1.append( A @ (U- Uhat))
    print()
    print(eps)
    print("truncation error")
    print(sl.norm(tau1[count], np.inf))

    count = count + 1
