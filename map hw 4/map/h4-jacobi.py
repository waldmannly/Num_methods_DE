import numpy as np
from numpy.linalg import *

def jacobi(A, b, x0, tol, maxiter=200):
    """
    Performs Jacobi iterations to solve the line system of
    equations, Ax=b, starting from an initial guess, ``x0``.

    Terminates when the change in x is less than ``tol``, or
    if ``maxiter`` [default=200] iterations have been exceeded.

    Returns 3 variables:
        1.  x, the estimated solution
        2.  rel_diff, the relative difference between last 2
            iterations for x
        3.  k, the number of iterations used.  If k=maxiter,
            then the required tolerance was not met.
    """
    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    k = 0
    rel_diff = tol * 2

    while (rel_diff > tol) and (k < maxiter):

        for i in range(0, n):
            subs = 0.0
            for j in range(0, n):
                if i != j:
                    subs += A[i,j] * x_prev[j]

            x[i] = (b[i] - subs ) / A[i,i]
        k += 1

        rel_diff = norm(x - x_prev) / norm(x)
        print(x, rel_diff)
        x_prev = x.copy()

    return x, rel_diff, k

# Main code starts here
# ---------------------
GL = 1.6
d = 0.8
A = np.array([
    [1.0,      0,      0,      0,   0],
    [ GL, -(d+1),    1.0,      0,   0],
    [  0,      d, -(d+1),    1.0,   0],
    [  0,      0,      d, -(d+1), 1.0],
    [  0,      0,      0,      0, 1.0]])
b = [0.5,      0,      0,      0, 0.1]
x0 = np.zeros(5);

tol = 1E-9
maxiter = 200
x, rel_diff, k = jacobi(A, b, x0, tol, maxiter)
if k == maxiter:
    print(('WARNING: the Jacobi iterations did not '
           'converge within the required tolerance.'))
print(('The solution is %s; within a tolerance of %g, '
        'using %d iterations.' % (x, rel_diff, k)))
print('Solution error = norm(Ax-b) = %g' % \
            norm(np.dot(A,x)-b))
print('Condition number of A = %0.5f' % cond(A))
print('Solution from built-in functions = %s' % solve(A, b))
