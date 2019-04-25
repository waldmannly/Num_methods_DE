import numpy as np
import scipy.sparse as sp
from grid import *
from problem import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# PART 2
grid = make_grid_heatsink(4, 4)
A, f = assemble(grid)

trapezoid = Trapezoid(A, f, dt=0.01)

steps =[]
steps.append(np.ones(len(grid.connect)))
for x in np.arange(0,100) :
    steps.append(trapezoid.step(  steps[x] ) )
    if (x%25 == 0): # plot
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(grid.points[:,0], grid.points[:,1],steps[x-1] , triangles=triangulate(grid), cmap='binary', linewidths=0.2);
        plt.show()


# PART 3
for i in [2,4,8]:
    steps=[]
    grid = make_grid_heatsink(4, i)
    val = np.ones(len( grid.connect ) )
    A, f = assemble(grid)
    print("heat flux " + str(i) + " fins")
    print(  heat_flux_south(grid,spa.spsolve(sp.csc_matrix(A),f) ) )
    trapezoid = Trapezoid(A, f, dt=0.01)
    for x in np.arange(0,100):
        val = trapezoid.step( val )
        steps.append( heat_flux_south(grid,val ))
    plt.figure()
    plt.plot(np.arange(0,100), steps)
    plt.title(i)
    plt.show()
