"""Unit-tests for the tomo_output module.

Run as python test_tomo_output.py in console or via coverage
"""

import os
import shutil
import unittest

import numpy as np
import numpy.testing as nptest

import tomo.utils.tomo_output as tout

base_dir = os.path.split(os.path.realpath(__file__))[0]
base_dir = os.path.split(base_dir)[0]
tmp_dir = os.path.join(base_dir, 'tmp')

# All values retrieved from INDIVShavingC325.dat
MACHINE_ARGS = {
    'output_dir':          '/tmp/',
    'dtbin':               9.999999999999999E-10,
    'dturns':              5,
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


class TestTomoOut(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    def test_create_phase_space_image(self):

        weights = np.ones(25)

        ax = np.arange(5)
        xp, yp = np.meshgrid(ax, ax)
        xp = np.vstack(xp.flatten())
        yp = np.vstack(yp.flatten())

        nbins = 5
        recprof = 0
        img = tout.create_phase_space_image(xp, yp, weights, nbins, recprof)

        correct = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04]])

        nptest.assert_equal(
            img, correct, err_msg='Phase space image was created incorrectly')
