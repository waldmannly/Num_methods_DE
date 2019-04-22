import numpy as np
import scipy.sparse as sp
from grid import *


def assemble(grid):
    """
    Assembles the matrix A and right hand side F for the spacial discreization

    grid:

        Defines a grid. Usually this grid is the return value of the function
        make_grid_heatsink in grid.py

    returns:

        A Scipy sparse matrix A and a numpy vector F containing the matrix and
        right hand side of the spacial discretization.

        Use a second order one-sided finite difference formula for the Neumann
        boundary conditions. For simplicity, at all upper corners we choose the
        outward normal `n` to be vertical.

        At the lower right and left corner the Dirichlet and Neumann boundary
        conditions meet. Choose the Neumann conditions here, with an outward
        unit vector `n` in horizontal direction.

    """
    def f(x): return 0

    print(grid)
    print(grid.points)
    print("grid h")
    print(grid.h)
    print("grid connect ")
    print(grid.connect)
    k=1
    kthrow = grid.connect[1]
    print(kthrow)
    print(kthrow[0])
    print(grid.points[kthrow[0]]) # this is the x and y


    print("loop start")
    mat = np.zeros((len(grid.connect), 2))
    F = np.zeros((len(grid.connect)))
    # loop through the k rows of A
    for k in np.arange(0,len(grid.connect)) :
        rowK =grid.connect[k]
        lapcian = -4*grid.points[k] # 4 times current point
        F[k] = f(grid.points[k])
        # then we loop from 0 to 3 for each direction
        for dir in [0,1,2,3]:
            # take the grid points of each of these directions
            if(rowK[dir] != None):
                XYval = grid.points[rowK[dir]]
                # make lapcian equation 4*current + 1* each other direction
                lapcian = lapcian + XYval
        mat[k][0]= lapcian[0]
        mat[k][1]= lapcian[1]
    print("end loop")
    print(mat*4)

    # for the unit cube..
    print("trying the book way")
    A = np.zeros((9,9))
    F = np.zeros((len(grid.connect)))
    for k in np.arange(0,len(grid.connect)) :
        rowK =grid.connect[k]
        A[k,k] = A[k,k] -4
        for dir in [0,1,2,3]:
            if(rowK[dir] != None):# if not none then fill in value
                A[k,rowK[dir]] = A[k,rowK[dir]] +1
            else:     # add in boundary conditions if none
                if (dir == 2): # south boundary conditions
                    F[k] = F[k] +1
                else: # all other boundary conditions
                    F[k] = F[k] - .5
    print(  A)


    print("adding in the project second order finite difference")
    A = np.zeros((9,9))
    F = np.zeros((len(grid.connect)))
    for k in np.arange(0,len(grid.connect)) :
        rowK =grid.connect[k]
        A[k,k] = A[k,k] - 3/2
        if (k+1 < len(grid.connect)):
            A[k,k+1] = A[k,k+1] + 2
        # else # add to F
        if (k+2 < len(grid.connect)):
            A[k,k+2] = A[k,k+2] - 1/2
        # else # add to F
    print(  A)

    print("adding in the project second order finite difference")
    A = np.zeros((9,9))
    F = np.zeros((len(grid.connect)))
    for k in np.arange(0,len(grid.connect)) :
        rowK =grid.connect[k]
        A[k,k] = A[k,k] - 3/2
        if (k-1 >= 0):
            A[k,k-1] = A[k,k-1] + 2
        # else # add to F
        if (k-2 >= 0):
            A[k,k-2] = A[k,k-2] - 1/2
        # else # add to F

    print(  A)


    # fourpointmethod = 1/12/h/h *(-1*U[j-2] + 16*U[j-1] - 30*U[j] + 16*U[j+1] - U[j-2])
    # oneside2ndorderD = -3/2*U[j] +2*U[j+1] -1/2*U[j+2]

    #the column number wherethe kth row of the A matrix... in the east direction? is given by the below
    # connect(k, east)
    #if none is in the connect then we are on the boundary

    # you give connect a point, k, and then a direction and it gives you the number of the point that you need

    # f[9] - f(grid.points(9))
    # where gridpoints gives you the x and y values
    print("F")
    print(F)

    pass


def heat_flux_south(grid, u):
    """
    Approximates the heat flux thourgh the southern boundary.

    grid

        Defines a grid. Usually this grid is the return value of the function
        make_grid_heatsink in grid.py

    u

        Numpy array, containing the soluton values at the grid points

    returns

        The heat flux through the southern boundary, a scalar.

    """
    pass


class Trapezoid:
    """
    Implementation of the trapezoid method for the linear ODE

        U' + AU = F

    """
    def __init__(self, A, F, dt):
        """

        A, F

            Matrix and vector defining the ODE.

        dt

            Time step size of the trapezoid method.

        """
        pass

    def step(self, U):
        """
        Computes one step of the trapezoid method.

        U

            Numpy vector, the initial value.

        returns

            The solution after one timestep starting from U.


        """
        pass

grid = make_grid_square(2)
print(assemble(grid))
