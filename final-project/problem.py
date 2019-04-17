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
    print(grid)
    print(grid.points)
    #the column number wherethe kth row of the A matrix... in the east direction? is given by the below
    # connect(k, east)
    #if none is in the connect then we are on the boundary

    # you give connect a point, k, and then a direction and it gives you the number of the point that you need

    # f[9] - f(grid.points(9))
    # where gridpoints gives you the x and y values


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
