# Evan Waldmann
# 2-24-19
# 3.2 Homework

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import math
from scipy.sparse import csr_matrix
import time

# part 1
n = 89*89
n=120
h= 1/n
k = np.array([np.ones(n-1),-2*np.ones(n),np.ones(n-1)]) #2nd order central
offset = [-1,0,1]
A = csr_matrix(-1* n*n * sp.diags(k,offset))
#I =csr_matrix( np.eye(n))
#S = sp.bmat([[A,I], [I,A]])

# D = np.copy(A.todense())# I got a memory error beyond
D = np.ones((n*n,n*n))
print(D)

#part 2
sizeA = (A.data.nbytes + A.indptr.nbytes + A.indices.nbytes )
print("Size of A and D in megabytes, respectively: ")
print(sizeA*1e-6)
print(D.data.nbytes*1e-6)

#part 3
v = np.ones(n)
t1= time.time()
a1= v@A
t2= time.time()
print("Sparse time :")
print(t2-t1)
t3= time.time()
d1= v@D
t4= time.time()
print("Dense time :")
print(t4-t3)

# print()
# print()
# print()
# # timing at different m's (part 4)
# T1=[]
# T2=[]
# ms =  np.arange(10, 251, 10)
# for m in ms :
#     n = m
#     h= 1/n
#
#     k = np.array([np.ones(n-1),-2*np.ones(n),np.ones(n-1)]) #2nd order central
#     offset = [-1,0,1]
#     A = csr_matrix(-1* n*n * sp.diags(k,offset))
#     # print(A)
#
#     D = np.copy(A.todense())
#     # print(D)
#
#     sizeA = (A.data.nbytes + A.indptr.nbytes + A.indices.nbytes )
#
#     print("Size of A and D in megabyte, respectively: ")
#     print(sizeA*1e-6)
#     print(D.data.nbytes*1e-6)
#
#
#     v = np.ones(n)
#     t1= time.time()
#     for i in np.arange(1,10000):
#         a1= v@A
#     t2= time.time()
#     print("Sparse time (for 10000 multiplications):")
#     print(t2-t1)
#     T1.append(t2-t1)
#
#     t3= time.time()
#     for i in np.arange(1,10000):
#         d1= v@D
#     t4= time.time()
#     print("Dense time (for 10000 multiplications):")
#     print(t4-t3)
#     T2.append(t4-t3)
#
# plt.figure()
# plt.plot(ms, T1, 'b', lw=2, label='Sparse Time')
# plt.plot(ms, T2, 'r', lw=2, label='Dense Time')
# plt.legend(loc='best')
# plt.show()
