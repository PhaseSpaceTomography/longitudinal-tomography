import unittest
import numpy as np
import numpy.testing as nptest
import ctypes
import os
from unit_tests.C500values import C500
from tomo.cpp_routines.tomolib_wrappers import back_project
from tomo.tomography.tomography_cpp import TomographyCpp
from tomo.tomography.tomography_py import TomographyPy

class TestTomographyCpp(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.resources_path = os.path.realpath(__file__)
        cls.resources_path = '/'.join(cls.resources_path.split('/')[:-1])
        cls.resources_path += '/resources'
        cls.c500 = C500()

    
    def test_reciprocal_paricles(self):
        nparts = 10
        nprofs = 5
        nbins = 5
        
        profiles = np.ones((nbins, nprofs))
        xp = np.zeros((nparts, profiles.shape[0]))
        
        for i in range(nprofs):
            for j in range(nparts):
                xp[j, i] = i
        xp[3, :] = 1
        xp[7, 3] = nbins - 1
        xp = xp.astype(np.int32)
        
        tomo = TomographyCpp(profiles, xp)
        ans = tomo._reciprocal_particles()
        
        right_ans = np.ones((nbins, nprofs)) * 10
        right_ans[0, 0] = 1.1111111111111112
        right_ans[1, 1] = 1.0
        right_ans[2, 2] = 1.1111111111111112
        right_ans[3, 3] = 1.25
        right_ans[4, 4] = 1.1111111111111112
        
        nptest.assert_almost_equal(
            ans, right_ans, err_msg='The reciprocal of the nr of '
                                    'the number of particles pr bin '
                                    'were calculated incorrectly')

    def test_create_flat_points(self):
        nparts = 4
        nprofs = 4
        nbins = 7

        profiles = np.ones((nprofs, nbins))
        xp = np.ones((nparts, nprofs))

        tomo = TomographyCpp(profiles, xp)

        flat_xp = tomo._create_flat_points()

        correct_ans = np.zeros(xp.shape, dtype=int)
        correct_ans[:] = np.arange(1, 23, 7)
        
        nptest.assert_equal(
            flat_xp, correct_ans, err_msg='The flattened xp values were '
                                          'calculated incorrectly')

    def test_discrepancy(self):
        profile1 = np.array([[0.79541, 0.238714, 3.4771101, 3.7630594, 1.374539,
                              2.3439029, 2.313675, 3.4513814,0.38265, 2.0089653],
                             [2.189960, 1.231416, 1.015321, 3.759437, 2.246436,
                              2.128640, 0.538299, 4.685674, 2.511942, 4.880746]])
        profile2 = np.array([[2.189960, 1.231416, 1.015321, 3.759437, 2.246436,
                              2.128640, 0.538299, 4.685674, 2.511942, 4.880746],
                              [0.79541, 0.238714, 3.4771101, 3.7630594, 1.374539,
                              2.3439029, 2.313675, 3.4513814, 0.38265, 2.0089653]])
        diff = profile1 - profile2

        dummy_xp = np.array([0])

        tomo = TomographyCpp(profile1, dummy_xp)
        ans = tomo._discrepancy(diff)

        correct_ans = 1.6508675030612614

        self.assertAlmostEqual(
            ans, correct_ans, msg='Discrepancy was calculated incorrectly')

    def test_project(self):
        nbins = 5
        nparts = 5
        nprofs = 10
        weight = np.arange(1, nparts + 1)
        flat_pts = np.array([[0,1,4,4,4,1,1,1,0,3],
                             [1,1,3,1,1,1,4,2,3,1],
                             [2,1,4,2,2,3,4,2,4,0],
                             [2,2,1,0,3,2,4,3,0,3],
                             [3,4,2,2,0,3,3,4,3,0]])

        dummy_profs = np.ones((nprofs, nbins))
        tomo = TomographyCpp(dummy_profs, flat_pts)

        for i in range(nprofs):
            flat_pts[:, i] += nbins * i
    
        weight = np.ascontiguousarray(weight).astype(ctypes.c_double)
        flat_pts = np.ascontiguousarray(flat_pts).astype(ctypes.c_int)

        recreated = tomo.project(flat_pts, weight)

        correct_ans = np.array([
            [0.06666667, 0.13333333, 0.46666667, 0.33333333, 0.        ],
            [0.        , 0.4       , 0.26666667, 0.        , 0.33333333],
            [0.        , 0.26666667, 0.33333333, 0.13333333, 0.26666667],
            [0.26666667, 0.13333333, 0.53333333, 0.        , 0.06666667],
            [0.33333333, 0.13333333, 0.2       , 0.26666667, 0.06666667],
            [0.        , 0.2       , 0.26666667, 0.53333333, 0.        ],
            [0.        , 0.06666667, 0.        , 0.33333333, 0.6       ],
            [0.        , 0.06666667, 0.33333333, 0.26666667, 0.33333333],
            [0.33333333, 0.        , 0.        , 0.46666667, 0.2       ],
            [0.53333333, 0.13333333, 0.        , 0.33333333, 0.        ]])

        nptest.assert_almost_equal(
            recreated, correct_ans,
            err_msg='Error when projecting to time space')

    # Tests back project straight from the tomolib_wrappers module,
    #  without going through TomographyCpp
    def test_back_project(self):
        nbins = 5
        nparts = 5
        nprofs = 10
        profiles = np.array([
            [0.06666667, 0.13333333, 0.46666667, 0.33333333, 0.        ],
            [0.        , 0.4       , 0.26666667, 0.        , 0.33333333],
            [0.        , 0.26666667, 0.33333333, 0.13333333, 0.26666667],
            [0.26666667, 0.13333333, 0.53333333, 0.        , 0.06666667],
            [0.33333333, 0.13333333, 0.2       , 0.26666667, 0.06666667],
            [0.        , 0.2       , 0.26666667, 0.53333333, 0.        ],
            [0.        , 0.06666667, 0.        , 0.33333333, 0.6       ],
            [0.        , 0.06666667, 0.33333333, 0.26666667, 0.33333333],
            [0.33333333, 0.        , 0.        , 0.46666667, 0.2       ],
            [0.53333333, 0.13333333, 0.        , 0.33333333, 0.        ]])

        flat_pts = np.array([[0,1,4,4,4,1,1,1,0,3],
                             [1,1,3,1,1,1,4,2,3,1],
                             [2,1,4,2,2,3,4,2,4,0],
                             [2,2,1,0,3,2,4,3,0,3],
                             [3,4,2,2,0,3,3,4,3,0]])
        for i in range(nprofs):
            flat_pts[:, i] += nbins * i

        weight = np.zeros(nparts)

        weight = np.ascontiguousarray(weight).astype(ctypes.c_double)
        flat_pts = np.ascontiguousarray(flat_pts).astype(ctypes.c_int)
        flat_profs = np.ascontiguousarray(
                        profiles.flatten()).astype(ctypes.c_double)

        weight = back_project(weight, flat_pts, flat_profs, nparts, nprofs)
        
        correct_ans = [1.8666666799999998, 2.6666666500000002,
                       4.066666660000001, 3.3333333499999993,
                       4.066666639999999]

        nptest.assert_almost_equal(weight, correct_ans,
                                   err_msg='Error in calculation of '\
                                           'back_projection')
