'''Unit-tests for the Profiles class.

Run as python test_tracking.py in console or via coverage
'''

import numpy as np
import numpy.testing as nptest
import os
import scipy.signal as sig
import unittest

import tomo.utils.exceptions as expt
import tomo.tracking.machine as mch
import tomo.data.profiles as prf


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


class TestProfiles(unittest.TestCase):

    def test_create_waterfall_no_iter_fails(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()
        waterfall = 1

        with self.assertRaises(expt.WaterfallError,
                               msg='Waterfall as non-iterable should '
                                   'raise an exception'):
            profiles = prf.Profiles(machine, machine.dtbin, waterfall)

    def test_create_waterfall_wrong_shape_fails(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()
        waterfall = np.ones((30, 100))

        with self.assertRaises(expt.WaterfallError,
                               msg='Waterfall as having the wrong amount '
                                   'of profiles should raise an exception'):
            profiles = prf.Profiles(machine, machine.dtbin, waterfall)

    def test_create_waterfall_reduced_to_zeros_fail(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()
        waterfall = np.ones((150, 100)) * -1

        with self.assertRaises(expt.WaterfallReducedToZero,
                               msg='Waterfall as having the wrong amount '
                                   'of profiles should raise an exception'):
            profiles = prf.Profiles(machine, machine.dtbin, waterfall)

    def test_calc_profile_charge_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()
        waterfall = self._load_waterfall()

        profiles = prf.Profiles(machine, machine.dtbin, waterfall)
        profiles.calc_profilecharge()

        # Move decimal to improve comparal
        correct_prof_charge = 206096981027.60077
        self.assertEqual(
            (profiles.profile_charge / 100000), (correct_prof_charge / 100000),
            msg='The profile charge was calculated incorrectly')

    def test_calc_self_fields_no_prof_charge_fails(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()
        waterfall = self._load_waterfall()
        
        profiles = prf.Profiles(machine, machine.dtbin, waterfall)
        with self.assertRaises(expt.ProfileChargeNotCalculated,
                               msg='An exception should be raised when '
                                   'an attempt is made to calculate the '
                                   'self-fields of a bunch, whitout first '
                                   'having provided the profile charge.'):
            profiles.calc_self_fields()

    def test_calc_self_fields_original_correct(self):
        waterfall = self._load_waterfall()
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        sample_time = machine.dtbin 

        # Update fields due to loading of rebinned waterfall. 
        rbn = 3
        machine.dtbin *= rbn
        machine.synch_part_x /= rbn

        # Set space charge parameters
        machine.g_coupling = 1.0
        machine.zwall_over_n = 50.0

        profiles = prf.Profiles(machine, sample_time, waterfall)
        profiles.calc_profilecharge()
        profiles.calc_self_fields()

        correct_vself = self._load_vself()
        correct_phiwrap = 6.283185307179586
        correct_wrap_length = 455

        self.assertAlmostEqual(profiles.phiwrap, correct_phiwrap,
                               msg='phiwrap calculated incorrectly.')
        self.assertAlmostEqual(profiles.wrap_length, correct_wrap_length,
                               msg='wrap length calculated incorrectly.')
        nptest.assert_almost_equal(
            profiles.vself, correct_vself,
            err_msg='Error in calculation self fields using original '
                    'method and filter.')

    def test_calc_self_fields_filter_outside_correct(self):
        waterfall = self._load_waterfall()
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        sample_time = machine.dtbin

        # Update fields due to loading of rebinned waterfall. 
        rbn = 3
        machine.dtbin *= rbn
        machine.synch_part_x /= rbn

        # Set space charge parameters
        machine.g_coupling = 1.0
        machine.zwall_over_n = 50.0

        profiles = prf.Profiles(machine, sample_time, waterfall)
        profiles.calc_profilecharge()
        
        smoothed_profs = np.copy(waterfall)
        smoothed_profs /= np.sum(smoothed_profs, axis=1)[:, None]
        smoothed_profs = sig.savgol_filter(
                                x=smoothed_profs, window_length=7,
                                polyorder=4, deriv=1)

        profiles.calc_self_fields(filtered_profiles=smoothed_profs)


        correct_vself = self._load_vself()
        correct_phiwrap = 6.283185307179586
        correct_wrap_length = 455

        self.assertAlmostEqual(profiles.phiwrap, correct_phiwrap,
                               msg='phiwrap calculated incorrectly.')
        self.assertAlmostEqual(profiles.wrap_length, correct_wrap_length,
                               msg='wrap length calculated incorrectly.')
        nptest.assert_almost_equal(
            profiles.vself, correct_vself,
            err_msg='Error in calculation self fields using original '
                    'method, filtered profiles provided via parameters')

    def test_calc_self_fields_filter_outside_no_iter_fails(self):
        waterfall = self._load_waterfall()
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        sample_time = machine.dtbin

        profiles = prf.Profiles(machine, sample_time, waterfall)
        profiles.calc_profilecharge()     

        with self.assertRaises(
                expt.FilteredProfilesError,
                msg='Providing the filtered profiles as non-iterable object '
                    'should raise an exception'):
            profiles.calc_self_fields(filtered_profiles=1)

    def test_calc_self_fields_filter_outside_bad_shape_fails(self):
        waterfall = self._load_waterfall()
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        sample_time = machine.dtbin

        profiles = prf.Profiles(machine, sample_time, waterfall)
        profiles.calc_profilecharge()     

        with self.assertRaises(
                expt.FilteredProfilesError,
                msg='Providing the filtered profiles not having the '
                    'same shape as the waterfall should raise an exception'):
            profiles.calc_self_fields(filtered_profiles=np.ones((60, 40)))

    def _load_waterfall(self):
        base_dir = os.path.split(os.path.realpath(__file__))[0]
        base_dir = os.path.split(base_dir)[0]
        data_path = os.path.join(base_dir, 'resources')
    
        waterfall = np.load(os.path.join(
                        data_path, 'waterfall_INDIVShavingC325.npy'))
        return waterfall

    def _load_vself(self):
        base_dir = os.path.split(os.path.realpath(__file__))[0]
        base_dir = os.path.split(base_dir)[0]
        data_path = os.path.join(base_dir, 'resources')
        vself = np.load(os.path.join(
                        data_path, 'vself_INDIVShavingC325.npy'))
        return vself
