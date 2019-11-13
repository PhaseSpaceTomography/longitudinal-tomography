import os
import unittest
import numpy as np
import numpy.testing as nptest
from tomo.time_space import TimeSpace
from tomo.parameters import Parameters
from unit_tests.C500values import C500
from particles import Particles

resources_dir = os.path.realpath(__file__)
resources_dir = '/'.join(resources_dir.split('/')[:-1])
resources_dir += '/resources/particle_tracking'

class TestTrack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c500 = C500()

        input_file = cls.c500.path + 'C500MidPhaseNoise.dat'
        with open(input_file, 'r') as f:
            read = f.readlines()
        
        cls.raw_param = read[:98]
        cls.raw_data = np.array(read[98:], dtype=float)

    def setUp(self):
        par = Parameters()
        par.parse_from_txt(np.copy(self.raw_param))
        par.fill()
        self.timespace = TimeSpace(par)
        self.timespace.create(self.raw_data)
        

    def test_full_pp_distribution(self):
        self.timespace.par.full_pp_flag = True
        parts = Particles(self.timespace)

        parts.homogeneous_distribution()

        coorect_x = np.load(f'{resources_dir}/full_pp_C500_xp.npy')
        coorect_y = np.load(f'{resources_dir}/full_pp_C500_yp.npy')

        # The initial coordinates should be identical
        # independently of computer env. 
        nptest.assert_equal(parts.x_coords, coorect_x,
                            err_msg='Error in creation of '
                                    'initial x coordinates')
        nptest.assert_equal(parts.y_coords, coorect_y,
                            err_msg='Error in creation of '
                                    'initial y coordinates')

    def test_ijlimit_distribution(self):
        self.timespace.par.full_pp_flag = False
        parts = Particles(self.timespace)

        parts.homogeneous_distribution()

        coorect_x = np.load(f'{resources_dir}/C500_xp.npy')
        coorect_y = np.load(f'{resources_dir}/C500_yp.npy')

        # The initial coordinates should be identical
        # independently of computer env. 
        nptest.assert_equal(parts.x_coords, coorect_x,
                            err_msg='Error in creation of '
                                    'initial x coordinates')
        nptest.assert_equal(parts.y_coords, coorect_y,
                            err_msg='Error in creation of '
                                    'initial y coordinates')

    # Unequal length of coordinate arrays should raise an exception
    def test_set_coordinates(self):
        nparts = 10
        x_coords = np.ones(nparts)
        y_coords = np.ones(nparts+1)
        parts = Particles(self.timespace)
        with self.assertRaises(AssertionError):
            parts.set_coordinates(x_coords, y_coords)

        y_coords = np.ones(nparts)
        parts.set_coordinates(x_coords, y_coords)
        nptest.assert_equal(x_coords, parts.x_coords)
        nptest.assert_equal(y_coords, parts.y_coords)


    def test_inital_coords_to_physical(self):
        x_coords = np.array([100, 125, 150])
        y_coords = np.array([150, 125, 100])

        parts = Particles(self.timespace)
        parts.set_coordinates(x_coords, y_coords)

        dphi, denergy = parts.init_coords_to_physical(0)

        correct_dphi = np.array([0.204732477330, 0.631258471768,
                                 1.057784466206])
        correct_denergy = np.array([1108680.081225472735,
                                    525164.249001539662, -58351.583222393303])

        nptest.assert_almost_equal(dphi, correct_dphi)
        nptest.assert_almost_equal(denergy, correct_denergy)

    def test_physical_to_coords(self):
        dphi = np.array([[0.204732477330],
                         [0.631258471768],
                         [1.057784466206]])
        denergy = np.array([[1108680.081225472735],
                            [525164.249001539662],
                            [-58351.583222393303]])

        parts = Particles(self.timespace)
        xp, yp = parts.physical_to_coords(dphi, denergy)

        correct_xp = np.array([[99.999999999984],
                               [124.998818705739],
                               [149.996568913391]])

        correct_yp = np.array([[150.000000000000],
                               [125.000000000000],
                               [100.000000000000]])

        nptest.assert_almost_equal(xp, correct_xp)
        nptest.assert_almost_equal(yp, correct_yp)
