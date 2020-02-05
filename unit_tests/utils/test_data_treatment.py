'''Unit-tests for the physics module.

Run as python test_data_treament.py in console or via coverage
'''

import numpy as np
import numpy.testing as nptest
import os
import unittest

import tomo.tracking.machine as mch
import tomo.data.profiles as prf
import tomo.utils.data_treatment as treat


# Machine arguments mased on the input file INDIVShavingC325.dat
MACHINE_ARGS = {
    'output_dir':          '/tmp/',
    'dtbin':               9.999999999999999E-10,
    'dturns':              5,
    'synch_part_x':        334.00000000000006,
    'demax':               -1.E6,
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
    'max_dt':              9.999999999999999E-10 * 760 # dtbin * nbins
}


class TestDataTreatment(unittest.TestCase):

    def test_calc_baseline_ftn_correct(self):
        waterfall = self._load_waterfall()
        correct = 0.00014048793859649817
        ans = treat.calc_baseline_ftn(waterfall, ref_prof=0)
        self.assertAlmostEqual(
            ans, correct, msg='Baseline calculated incorrectly')

    def test_rebin_odd_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nbins = 9
        machine.synch_part_x = 4.53
        machine.dtbin = 0.535

        waterfall = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                              [10, 21, 32, 43, 54, 65, 76, 86, 97]],
                             dtype=float)
        rbn = 2
        rebinned = treat.rebin(waterfall, rbn, machine=machine)

        correct_rebinned = np.array([[3.0, 7.0, 11.0, 15.0, 16.0],
                                     [31.0, 75.0, 119.0, 162.0, 172.66666667]])
        
        nptest.assert_almost_equal(
            rebinned, correct_rebinned,
            err_msg='Rebinning array of odd length failed')

        # Checks that the x coordinate of synchronous particle
        # is updated to fit the new number of bins. 
        updated_synch_part_x = 2.265
        self.assertAlmostEqual(machine.synch_part_x, updated_synch_part_x,
                               msg='Error in updated synch part x')

        # Checks that the size of the bins are updated
        # is updated to fit the new number of bins.
        updated_dtbin = 1.07
        self.assertAlmostEqual(machine.dtbin, updated_dtbin,
                               msg='Error in updated dtbin')

    def test_rebin_even_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nbins = 8
        machine.synch_part_x = 4.53
        machine.dtbin = 0.535

        waterfall = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                              [10, 21, 32, 43, 54, 65, 76, 86]],
                             dtype=float)
        rbn = 2
        rebinned = treat.rebin(waterfall, rbn, machine=machine)

        correct_rebinned = np.array([[3.0, 7.0, 11.0, 15.0],
                                     [31.0, 75.0, 119.0, 162.0]])
        
        nptest.assert_almost_equal(
            rebinned, correct_rebinned,
            err_msg='Rebinning array of odd length failed')

        # Checks that the x coordinate of synchronous particle
        # is updated to fit the new number of bins. 
        updated_synch_part_x = 2.265
        self.assertAlmostEqual(machine.synch_part_x, updated_synch_part_x,
                               msg='Error in updated synch part x')

        # Checks that the size of the bins are updated
        # is updated to fit the new number of bins.
        updated_dtbin = 1.07
        self.assertAlmostEqual(machine.dtbin, updated_dtbin,
                               msg='Error in updated dtbin')

    def test_rebin_dtbin_update_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nbins = 8
        machine.synch_part_x = 4.53
        machine.dtbin = 0.535

        waterfall = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                              [10, 21, 32, 43, 54, 65, 76, 86]],
                             dtype=float)
        rbn = 2
        _, dtbin = treat.rebin(waterfall, rbn, dtbin=machine.dtbin)

        correct = 1.07
        self.assertEqual(dtbin, correct, msg='error in calculation of '
                                             'updated dtbin')


    def test_fit_synch_part_x_correct_x_coord(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.time_at_turn = np.array([1.3701195948153858e-06])
        machine.omega_rev0 = np.array([4585866.32214847])
        machine.phi0 = np.array([0.40078213])

        waterfall = self._load_waterfall()
        profile = prf.Profiles(machine, machine.dtbin, waterfall)

        correct_x = 134.08972093246575
        fitted_x, _, _ = treat.fit_synch_part_x(profile)

        self.assertAlmostEqual(
            fitted_x, correct_x, msg='Fitted x coordinate of synchronous '
                                     'particle is not correct' )
        
    def test_fit_synch_part_x_correct_tfoot_low(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.time_at_turn = np.array([1.3701195948153858e-06])
        machine.omega_rev0 = np.array([4585866.32214847])
        machine.phi0 = np.array([0.40078213])

        waterfall = self._load_waterfall()
        profile = prf.Profiles(machine, machine.dtbin, waterfall)

        correct_tfoot_low = 21.569078947368386
        _, tfoot_low, _ = treat.fit_synch_part_x(profile)
        
        self.assertAlmostEqual(
            tfoot_low, correct_tfoot_low,
            msg='Lower tangent foot is calculated incorrectly' )

    def test_fit_synch_part_x_correct_tfoot_up(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.time_at_turn = np.array([1.3701195948153858e-06])
        machine.omega_rev0 = np.array([4585866.32214847])
        machine.phi0 = np.array([0.40078213])

        waterfall = self._load_waterfall()
        profile = prf.Profiles(machine, machine.dtbin, waterfall)

        correct_tfoot_up = 255.84868421052698
        _, _, tfoot_up = treat.fit_synch_part_x(profile)
        
        self.assertAlmostEqual(
            tfoot_up, correct_tfoot_up,
            msg='Upper tangent foot is calculated incorrectly' )

    def _load_waterfall(self):
        base_dir = os.path.split(os.path.realpath(__file__))[0]
        base_dir = os.path.split(base_dir)[0]
        data_path = os.path.join(base_dir, 'resources')
    
        waterfall = np.load(os.path.join(
                        data_path, 'waterfall_INDIVShavingC325.npy'))
        return waterfall
