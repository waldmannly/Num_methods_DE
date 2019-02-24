import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa
import math

n1 = 100
n2 = 100
n3 = 100
n4 = 2
M1 = np.eye((n1))
M2 = np.eye((n2))
M3 = np.eye((n3))

def blockTri(Ms):
    #Takes in a list of matrices (not square) and returns a tridiagonal block matrix with zeros on the diagonal
    count = 0
    idx = []
    for M in Ms:
        #print(M.shape)
        count += M.shape[0]
        idx.append(count)
    count += Ms[-1].shape[-1]
    mat = np.zeros((count,count))
    count = 0
    for i, M in enumerate(Ms):
        mat[count:count+M.shape[0],idx[i]:idx[i]+M.shape[1]] = M
        count = count + M.shape[0]
    mat = mat + mat.T
    return mat

M = blockTri([M1, M2, M3])
# print(M)
A = np.eye(400)
print(M + A)
