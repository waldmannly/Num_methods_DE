import collections
import numpy as np


north = 0
east = 1
south = 2
west = 3

def make_connectivity(i, j, i_point, offset, in_domain, connectivity):

    if in_domain(i-1,j):
        connectivity[i_point, west] = i_point-1
        connectivity[i_point-1][east] = i_point

    if in_domain(i,j-1):
        connectivity[i_point, south] = i_point - offset
        connectivity[i_point-offset, north] = i_point

def make_grid_square(level):

    grid = collections.namedtuple('Grid', [])
    n = 2**level
    grid.h = 1/n
    n_points = (n-1)**2

    grid.points = np.empty((n_points, 2))
    grid.connect = np.empty((n_points, 4), dtype=np.object)
    
    def in_domain(i,j):
    
        return (i>0) and (j>0)
    
    i_point = 0
    for j in range(1, n):
    
        offset = n-1 
        for i in range(1, n):
            if in_domain(i,j): 
                grid.points[i_point] =[i*grid.h, j*grid.h]
                make_connectivity(i, j, i_point, offset, in_domain, grid.connect)
                i_point += 1
            else:
                if j == ny // 2:
                    offset -=1

    return grid


def make_grid_heatsink(level, n_fins):
    """
    Makes a grid of points for the heat sink with 2**level-1 points in the 
    vertical direction and n_fins fins on the top side. 

    The return value 

        grid = make_grid_heatsink(...)

    contains three attributes:

        grid.h: scalar
        
            The mesh size.

        grid.points: numpy array

            Array of shape (n,2) for the x and y values of n grid points, 
            so that e.g. grid.points[5,1] is the y value of the 5th point.

        grid.connect: numpy array

            Array of shape (n,4) that describes the relative position of 
            grid points. The first index is the number of the point and 
            the second one of four directions (0=north, 1=east, etc.
            as defined at the top of this file). The value is the index
            of the point in the given direction. 

            For example with 

                i = grid.connect[5, north] 

            the point grid.point[i] is the first point north of 
            grid.point[5]. If this point is outside of the domain, 
            i == None.

    """
    grid = collections.namedtuple('Grid', [])
    ny = 2**level
    nx = 2*2**level
    grid.h = 1/nx
    n_points = (nx-1)*(ny//2-1) + (nx//2) * (ny//2)

    grid.points = np.empty((n_points, 2))
    grid.connect = np.empty((n_points, 4), dtype=np.object)
    
    def in_domain(i,j):
    
        return not (j*grid.h >= 0.5*ny*grid.h and (i//(nx//n_fins)) % 2 == 0)\
               and (i>0)\
               and (j>0)
    
    i_point = 0
    for j in range(1, ny):
    
        offset = nx-1 if j <= ny//2 else nx//2
        for i in range(1, nx):
            if in_domain(i,j): 
                grid.points[i_point] =[i*grid.h, j*grid.h]
                make_connectivity(i, j, i_point, offset, in_domain, grid.connect)
                i_point += 1
            else:
                if j == ny // 2:
                    offset -=1

    return grid


def in_boundary(i, grid, direction):
    return grid.connect[i, direction] is None


def triangulate(grid):

    triangles = []
    for i, p in enumerate(grid.points):


        if (not in_boundary(i, grid, south)) and (not in_boundary(i, grid, west)):
            triangles.append([grid.connect[grid.connect[i, south], west],
                              grid.connect[i, south], i])
            triangles.append([grid.connect[grid.connect[i, south], west],
                              i, grid.connect[i, west]])

    triangles = np.asarray(triangles)
    return triangles

        
