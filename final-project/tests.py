import unittest
import hashlib
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from grid import make_grid_square, make_grid_heatsink
from problem import assemble, heat_flux_south, Trapezoid
# from plot import plot_grid

from contextlib import contextmanager

class ProblemTests(unittest.TestCase):

    def test_sparse(self):

        grid = make_grid_square(2)
        A, f = assemble(grid)

        self.assertTrue(isinstance(A, sp.spmatrix))


    def test_assemble_square_matrix(self):

        grid = make_grid_square(2)
        A, f = assemble(grid)

        expected = 1/grid.h * np.array(
                   ([[ 1.5,  -2. ,   0.5,   0. ,   0.,    0. ,   0. ,  0. ,  0. ],
                     [-4. ,  16. ,  -4. ,   0. ,  -4.,    0. ,   0. ,  0. ,  0. ],
                     [ 0.5,  -2. ,   1.5,   0. ,   0.,    0. ,   0. ,  0. ,  0. ],
                     [ 0. ,   0. ,   0. ,   1.5,  -2.,    0.5,   0. ,  0. ,  0. ],
                     [ 0. ,  -4. ,   0. ,  -4. ,  16.,   -4. ,   0. , -4. ,  0. ],
                     [ 0. ,   0. ,   0. ,   0.5,  -2.,    1.5,   0. ,  0. ,  0. ],
                     [ 0.5,   0. ,   0. ,  -2. ,   0.,    0. ,   1.5,  0. ,  0. ],
                     [ 0. ,   0.5,   0. ,   0. ,  -2.,    0. ,   0. ,  1.5,  0. ],
                     [ 0. ,   0. ,   0.5,   0. ,   0.,   -2. ,   0. ,  0. ,  1.5]]))

        np.testing.assert_almost_equal(A.toarray(), expected)

    def test_assemble_square_rhs(self):

        grid = make_grid_square(2)
        A, f = assemble(grid)

        expected = np.array([-0.5, 16.,  -0.5, -0.5,  0.,  -0.5, -0.5, -0.5, -0.5])
        np.testing.assert_almost_equal(f, expected)

    def test_assemble_heatsink_matrix(self):

        grid = make_grid_heatsink(3, 2)
        A, f = assemble(grid)

        indices = [(0, 0), (0, 1), (0, 2), (4, 3), (4, 4), (4, 5), (4, 19),
                   (20, 5), (20, 19), (20, 20), (20, 21), (20, 35), (70, 54),
                   (70, 62), (70, 70)]

        values = [24.0, -32.0, 8.0, -256.0, 1024.0, -256.0, -256.0, -256.0,
                  -256.0, 1024.0, -256.0, -256.0, 8.0, -32.0, 24.0]

        for i, index in enumerate(indices):
            self.assertAlmostEqual(A[index], values[i])

    def test_assemble_heatsink_rhs(self):

        grid = make_grid_heatsink(3, 2)
        A, f = assemble(grid)

        expected = [ -0.5, 256. , 256. , 256. , 256. , 256. , 256. , 256. , 256. , 256. , 256. , 256. ,
                    256. , 256. ,  -0.5,  -0.5,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,
                      0. ,   0. ,   0. ,   0. ,   0. ,  -0.5,  -0.5,  -0.5,  -0.5,  -0.5,  -0.5,  -0.5,
                     -0.5,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,  -0.5,  -0.5,   0. ,   0. ,
                      0. ,   0. ,   0. ,   0. ,  -0.5,  -0.5,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,
                     -0.5,  -0.5,   0. ,   0. ,   0. ,   0. ,   0. ,   0. ,  -0.5,  -0.5,  -0.5,  -0.5,
                     -0.5,  -0.5,  -0.5,  -0.5,  -0.5]

        np.testing.assert_almost_equal(f, expected)

    def test_heat_flux_south(self):

        grid = make_grid_heatsink(3, 2)
        A, f = assemble(grid)

        u = grid.points[:,1]
        self.assertAlmostEqual(heat_flux_south(grid, u), -0.9375)

    def test_trapezoid(self):

        A = sp.csc_matrix(np.array([3.0]))
        f = sp.csc_matrix(np.array([4.0]))
        u = sp.csc_matrix(np.array([7.0]))
        trapezoid = Trapezoid(A, f, dt=0.5)

        self.assertAlmostEqual(trapezoid.step(u)[0], 2.14285714)



if __name__ == '__main__':

    print()
    print('Fingerprint:', hashlib.md5(open(__file__, 'rb').read()).hexdigest())
    print()
    unittest.main(verbosity=2)
