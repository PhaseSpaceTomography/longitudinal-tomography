"""Unit-tests for the Profiles class.

Run as python test_tracking.py in console or via coverage
"""

import os
import unittest

import numpy as np
import numpy.testing as nptest
import scipy.signal as sig

from .. import commons
import tomo.data.profiles as prf
import tomo.tracking.machine as mch
from tomo import exceptions as expt

# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = commons.get_machine_args()


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
        correct_prof_charge = 206096981027.60077 * 1.0e-11
        profiles.profile_charge *= 1.0e-11

        self.assertAlmostEqual(
            profiles.profile_charge, correct_prof_charge,
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

    def test_calc_self_fields_original_correct_bdot_greater_than_zero(self):
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
        self.assertEqual(profiles.wrap_length, correct_wrap_length,
                         msg='wrap length calculated incorrectly.')
        nptest.assert_almost_equal(
            profiles.vself, correct_vself,
            err_msg='Error in calculation self fields using original '
                    'method and filter.')

    def test_calc_self_fields_wraplength_correct_bdot_less_than_zero(self):
        waterfall = self._load_waterfall()
        machine = mch.Machine(**MACHINE_ARGS)
        machine.bdot = 0.0
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
        correct_wrap_length = 457

        self.assertAlmostEqual(profiles.phiwrap, correct_phiwrap,
                               msg='phiwrap calculated incorrectly.')
        self.assertEqual(profiles.wrap_length, correct_wrap_length,
                         msg='wrap length calculated incorrectly.')

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
