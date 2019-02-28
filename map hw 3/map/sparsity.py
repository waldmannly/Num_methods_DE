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
import scipy.sparse
from scipy.sparse import coo_matrix, block_diag
np.set_printoptions(linewidth=132)

# part 1
n=110
h= 1/n

offset = [-1,0,1]
A = csr_matrix(-1* n*n * sp.diags([1,-4,1],offset, (n,n))  )

S= csr_matrix(sp.bmat([[A if i==j else np.eye(n) if abs(i-j)==1
              else None for i in range(n)] for j in range(n)]))

# print(S)

# D = np.copy(A.todense())# I got a memory error with this
D = np.ones((n*n,n*n))
# print(D)

print()

#part 2
sizeS = (S.data.nbytes + S.indptr.nbytes + S.indices.nbytes )
print("Size of S and D in megabytes, respectively: ")
print(sizeS*1e-6)
print(D.data.nbytes*1e-6)

print()
#part 3
v = np.ones(n*n)
t1= time.time()
a1= v@S
t2= time.time()
print("Sparse time :")
print(t2-t1)
t3= time.time()
d1= v@D
t4= time.time()
print("Dense time :")
print(t4-t3)
