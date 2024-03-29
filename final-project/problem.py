import numpy as np
import scipy.sparse as sp
from grid import *
import scipy.sparse.linalg as spa

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

    def testdirs(rowK):
        for dir in [0,1,3,2]:
            if(rowK[dir] == None):
                return dir
        return 100

    A = sp.lil_matrix((len(grid.connect),len(grid.connect)))
    F = np.zeros((len(grid.connect)))
    for k in np.arange(0,len(grid.connect)) :
        rowK =grid.connect[k]
        d = testdirs(rowK)
        F[k] = -.5;
        if (d != 100): # dont use lapcian
            #get which way we are pointing.
            if (d == 0):
                # go south twice
                A[k,k] += -3/2
                A[k, rowK[2] ] += 2
                A[k, grid.connect[rowK[2]][2] ] += -1/2
                # A[k,rowk[3]] += -1/2
            elif (d == 1):
                # go west twice
                A[k,k] += -3/2
                A[k, rowK[3] ] += 2
                A[k, grid.connect[rowK[3]][3] ] += -1/2
            elif (d ==3):
                # go east twice
                A[k,k] += -3/2
                A[k, rowK[1] ] += 2
                A[k, grid.connect[rowK[1]][1] ] += -1/2
            else: # this should mean the only NONE is to the south
                # this should just be the Dirichlet BC's
                A[k,k] = A[k,k] -4* 1/grid.h
                for dir in [0,1,3]:
                    A[k,rowK[dir]] = A[k,rowK[dir]] +1* 1/grid.h
                F[k] = 1/(grid.h*grid.h) *1
        else :  # use lapcian
            A[k,k] = A[k,k] -4* 1/grid.h
            for dir in [0,1,2,3]:
                A[k,rowK[dir]] = A[k,rowK[dir]] +1* 1/grid.h
            F[k] = 0

    # print(-A)

    # oneside2ndorderD = -3/2*U[j] +2*U[j+1] -1/2*U[j+2]

    #the column number wherethe kth row of the A matrix... in the east direction? is given by the below
    # connect(k, east)
    #if none is in the connect then we are on the boundary

    # you give connect a point, k, and then a direction and it gives you the number of the point that you need

    return ((-1/grid.h * A) , F)


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


    gammaD =[]
    for k in np.arange(0,len(grid.connect)):
        if (grid.connect[k][2] == None):
            gammaD.append(k)
    sum=0
    for k in gammaD:
        sum = sum + -1*(- 3/2*u[k] +2*u[grid.connect[k][0]] -1/2*u[grid.connect[grid.connect[k][0]][0]])
    return sum


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
        self.A =A
        self.F = F
        self.dt = dt
        self.left =  sp.eye((self.A).shape[0],(self.A).shape[1]) + self.dt/2* self.A
        pass

    def step(self, U):
        """
        Computes one step of the trapezoid method.

        U

            Numpy vector, the initial value.

        returns

            The solution after one timestep starting from U.


        """
        RHS = U + self.dt/2*(self.F - self.A@U + self.F )
        solved = spa.spsolve( self.left, RHS )

        return solved
