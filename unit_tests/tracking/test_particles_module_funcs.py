"""Unit-tests for the functions in the particles module.

Run as python test_particles_class.py in console or via coverage
"""

import unittest

import numpy as np

import tomo.tracking.machine as mch
import tomo.tracking.particles as pts
from tomo import exceptions as expt

# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = {
    'output_dir': '/tmp/',
    'dtbin': 9.999999999999999E-10,
    'dturns': 5,
    'synch_part_x':        334.00000000000006,
    'demax': -1.E6,
    'filmstart':           0,
    'filmstop':            1,
    'filmstep':            1,
    'niter':               20,
    'snpt':                4,
    'full_pp_flag':        False,
    'beam_ref_frame':      0,
    'machine_ref_frame':   0,
    'vrf1':                2637.197030932989,
    'vrf1dot':             0.0,
    'vrf2':                0.0,
    'vrf2dot':             0.0,
    'h_num':               1,
    'h_ratio':             2.0,
    'phi12':               0.4007821253666541,
    'b0':                  0.15722,
    'bdot':                0.7949999999999925,
    'mean_orbit_rad':      25.0,
    'bending_rad':         8.239,
    'trans_gamma':         4.1,
    'rest_energy':         0.93827231E9,
    'charge':              1,
    'self_field_flag':     False,
    'g_coupling':          0.0,
    'zwall_over_n':        0.0,
    'pickup_sensitivity':  0.36,
    'nprofiles':           150,
    'nbins':               760,
    'min_dt':              0.0,
    'max_dt':              9.999999999999999E-10 * 760  # dtbin * nbins
}


class TestParticlesMethods(unittest.TestCase):

    def test_filter_lost_correct(self):
        xp = np.array([[1, 2, 3, 4, 5, 6, 7],
                       [1, 2, 3, 4, 5, 6, 7],
                       [1, 2, 3, 3, 3, 4, 4],
                       [1, 2, 3, 3, 3, 4, 6],
                       [1, 2, 3, 3, 3, 4, 1],
                       [1, 2, 5, 3, 3, 4, 1]])
        yp = np.ones(xp.shape)

        img_width = 5
        xp, yp, nr_lost = pts.filter_lost(xp, yp, img_width)

        correct_xp = np.array([[1, 2, 4], [1, 2, 4], [1, 2, 3],
                               [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        correct_yp = np.ones((6, 3))

        for xvec, cxvec in zip(xp, correct_xp):
            for x, cx in zip(xvec, cxvec):
                self.assertEqual(x, cx, msg='Error in xp coordinates '
                                            'after filtering')

        for yvec, cyvec in zip(yp, correct_yp):
            for y, cy in zip(yvec, cyvec):
                self.assertEqual(y, cy, msg='Error in yp coordinates '
                                            'after filtering')

        self.assertEqual(nr_lost, 4, msg='Error in reported number of '
                                         'lost particles')

    def test_filter_lost_all_pts_lost_error(self):
        xp = np.array([[5, 6, 7], [5, 6, 7], [3, 4, 4],
                       [3, 4, 6], [3, 4, 1], [3, 4, 1]])
        yp = np.ones(xp.shape)

        img_width = 5

        with self.assertRaises(expt.InvalidParticleError,
                               msg='Removing all particles should '
                                   'raise an exception'):
            xp, yp, nr_lost = pts.filter_lost(xp, yp, img_width)

    def test_physical_to_coords_correct(self):
        phases = [[-0.24180582, 0.04498875, 0.33178332, 0.61857789],
                  [-0.24180582, 0.04498875, 0.33178332, 0.61857789]]
        energies = [[-115567.32591061, -115567.32591061, -115567.32591061,
                     -115567.32591061],
                    [-38522.4419702,   -38522.4419702, -38522.4419702,
                     -38522.4419702]]

        phases = np.array(phases)
        energies = np.array(energies)

        machine = mch.Machine(**MACHINE_ARGS)

        machine.dturns = 5
        machine.phi0 = np.array([0.40078213, 0.40078213, 0.40078213,
                                 0.40078213, 0.40078213, 0.40078213])
        machine.h_num = 1
        machine.omega_rev0 = np.array([4585866.32214847, 4585893.44719289,
                                       4585920.57199441, 4585947.69655305,
                                       4585974.82086881, 4586001.94494169])
        machine.dtbin = 9.999999999999999e-10
        machine.synch_part_y = 380.0

        xorigin = -246.60492626420734
        dEbin = 1232.7181430465346

        xp, yp = pts.physical_to_coords(
            phases, energies, machine, xorigin, dEbin)

        correct_xp = [[281.27150807, 343.8103066,  406.34910513, 468.88790365],
                      [281.27048287, 343.80743192, 406.34438098, 468.88133003]]
        correct_yp = [[286.25, 286.25, 286.25, 286.25],
                      [348.75, 348.75, 348.75, 348.75]]

        for xvec, cxvec in zip(xp, correct_xp):
            for x, cx in zip(xvec, cxvec):
                self.assertAlmostEqual(
                    x, cx, msg='Error in calculated xp coordinates ')

        for yvec, cyvec in zip(yp, correct_yp):
            for y, cy in zip(yvec, cyvec):
                self.assertAlmostEqual(
                    y, cy, msg='Error in calculated yp coordinates ')

    def test_physical_to_coords_error(self):
        phases = [[-0.24180582, 0.04498875, 0.33178332, 0.61857789],
                  [-0.24180582, 0.04498875, 0.33178332, 0.61857789]]
        energies = [[-115567.32591061, -115567.32591061, -115567.32591061],
                    [-38522.4419702, -38522.4419702, -38522.4419702]]

        phases = np.array(phases)
        energies = np.array(energies)

        machine = mch.Machine(**MACHINE_ARGS)

        machine.dturns = 5
        machine.phi0 = np.array([0.40078213, 0.40078213, 0.40078213,
                                 0.40078213, 0.40078213, 0.40078213])
        machine.h_num = 1
        machine.omega_rev0 = np.array([4585866.32214847, 4585893.44719289,
                                       4585920.57199441, 4585947.69655305,
                                       4585974.82086881, 4586001.94494169])
        machine.dtbin = 9.999999999999999e-10
        machine.synch_part_y = 380.0

        xorigin = -246.60492626420734
        dEbin = 1232.7181430465346

        with self.assertRaises(expt.InvalidParticleError,
                               msg='Providing arrays of coordinates of '
                                   'different lengths should raise '
                                   'an exception'):
            xp, yp = pts.physical_to_coords(
                phases, energies, machine, xorigin, dEbin)
