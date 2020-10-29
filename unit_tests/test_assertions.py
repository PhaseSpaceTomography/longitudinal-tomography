"""Unit-tests for the assertions module.

Run as python test_assertions.py in console or via coverage
"""

import unittest

import numpy as np

import tomo.tracking.machine as mch
from tomo import assertions as asrt, exceptions as expt
import tomo.utils.tomo_input as tomoin

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
    'max_dt':              9.999999999999999E-10 * 760
}


class TestAssertions(unittest.TestCase):

    def test_assert_greater(self):
        var_name = 'test'
        limit = 15
        error_class = AssertionError

        var = 10
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_greater(var, var_name, limit, error_class)
        
        var = limit
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_greater(var, var_name, limit, error_class)
        
        var = 20
        asrt.assert_greater(var, var_name, limit, error_class)

    def test_assert_less(self):
        var_name = 'test'
        limit = 15
        error_class = AssertionError

        var = 20
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_less(var, var_name, limit, error_class)
        
        var = limit
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_less(var, var_name, limit, error_class)
        
        var = 10
        asrt.assert_less(var, var_name, limit, error_class)

    def test_assert_equal(self):
        var_name = 'test'
        limit = 15
        error_class = AssertionError

        var = 10
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_equal(var, var_name, limit, error_class)
        
        var = 20
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_equal(var, var_name, limit, error_class)
        
        var = limit
        asrt.assert_equal(var, var_name, limit, error_class)

    def test_assert_not_equal(self):
        var_name = 'test'
        limit = 15
        error_class = AssertionError

        var = 10
        asrt.assert_not_equal(var, var_name, limit, error_class)
        
        var = limit
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_not_equal(var, var_name, limit, error_class)
        
        var = 20
        asrt.assert_not_equal(var, var_name, limit, error_class)

    def test_assert_less_or_equal(self):
        var_name = 'test'
        limit = 15
        error_class = AssertionError

        var = 10
        asrt.assert_less_or_equal(var, var_name, limit, error_class)
        
        var = limit
        asrt.assert_less_or_equal(var, var_name, limit, error_class)

        var = 20
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_less_or_equal(var, var_name, limit, error_class)

    def test_assert_greater_or_equal(self):
        var_name = 'test'
        limit = 15
        error_class = AssertionError

        var = 10
        asrt.assert_less_or_equal(var, var_name, limit, error_class)
        
        var = limit
        asrt.assert_less_or_equal(var, var_name, limit, error_class)

        var = 20
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_less_or_equal(var, var_name, limit, error_class)

    def test_assert_inrange(self):
        var_name = 'test'
        low_lim = 15
        up_lim = 20
        error_class = AssertionError

        var = low_lim
        asrt.assert_inrange(var, var_name, low_lim, up_lim, error_class)

        var = up_lim
        asrt.assert_inrange(var, var_name, low_lim, up_lim, error_class)

        var = 17
        asrt.assert_inrange(var, var_name, low_lim, up_lim, error_class)

        var = 10
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_inrange(var, var_name, low_lim, up_lim, error_class)

        var = 25
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_inrange(var, var_name, low_lim, up_lim, error_class)

    def test_assert_array_not_equal(self):
        array = np.ones((10, 10))
        array_name = 'test_array'
        limit = 1
        error_class = AssertionError

        with self.assertRaises(
                    error_class, msg='An error should have been raised'):
            asrt.assert_array_not_equal(array, array_name, limit, error_class)

        array[1, 1] = 2
        asrt.assert_array_not_equal(array, array_name, limit, error_class)

    def test_assert_array_shape_equal(self):
        a_array = np.zeros((2, 4))        
        b_array = np.zeros((2, 4))
        c_array = np.zeros((7, 4))
        correct_shape = (2, 4)

        with self.assertRaises(
                expt.UnequalArrayShapes,
                msg='An error should have been raised'):
            asrt.assert_array_shape_equal(
                    (a_array, b_array, c_array),
                    ('a', 'b', 'c'), correct_shape)
        
        asrt.assert_array_shape_equal((a_array, b_array),
                                      ('a', 'b'), correct_shape)

    def test_assert_array_in_range(self):
        low_lim = 5
        up_lim = 15
        error_class = AssertionError

        array = np.arange(1, 20)

        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_array_in_range(array, low_lim, up_lim, error_class)

        array = np.arange(low_lim, up_lim+1)
        asrt.assert_array_in_range(array, low_lim, up_lim, error_class)        

    def test_assert_array_greater(self):
        limit = 10
        error_class = AssertionError
        
        array = np.ones(10) * 15
        asrt.assert_array_greater(array, limit, error_class)

        array[3] = limit
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_array_greater(array, limit, error_class)

        array[3] = 9
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_array_greater(array, limit, error_class)

    def test_assert_array_greater_or_equal(self):
        limit = 10
        error_class = AssertionError
        
        array = np.ones(10) * 15
        asrt.assert_array_greater_eq(array, limit, error_class)

        array[3] = limit
        asrt.assert_array_greater_eq(array, limit, error_class)

        array[3] = 9
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_array_greater_eq(array, limit, error_class)

    def test_assert_array_less(self):
        limit = 10
        error_class = AssertionError
        
        array = np.ones(10) * 5
        asrt.assert_array_less(array, limit, error_class)

        array[3] = limit
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_array_less(array, limit, error_class)

        array[3] = 11
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_array_less(array, limit, error_class)

    def test_assert_array_less_or_equal(self):
        limit = 10
        error_class = AssertionError
        
        array = np.ones(10) * 5
        asrt.assert_array_less_eq(array, limit, error_class)

        array[3] = limit
        asrt.assert_array_less_eq(array, limit, error_class)

        array[3] = 11
        with self.assertRaises(
                error_class, msg='An error should have been raised'):
            asrt.assert_array_less_eq(array, limit, error_class)

    def test_assert_only_valid_particles(self):
        nbins = 25
        particles = np.ones((10, 10))
        
        # Testing too large x coordinate
        particles[6, 9] = nbins
        with self.assertRaises(
                expt.InvalidParticleError,
                msg='Particles outside of the image width should raise an '
                    'exception.'):
            asrt.assert_only_valid_particles(particles, nbins)

        particles[6, 9] = -1
        with self.assertRaises(
                expt.InvalidParticleError,
                msg='Particles outside of the image width should raise an '
                    'exception.'):
            asrt.assert_only_valid_particles(particles, nbins)

    def test_assert_fields(self):
        error_class = AssertionError
        needed_fields = ['hope']
        machine = mch.Machine(**MACHINE_ARGS)
        
        with self.assertRaises(error_class,
                               msg='Object lacking needed fields '
                                   'should raise an exception'):
            asrt.assert_fields(
                machine, 'test_machine', needed_fields, error_class)

    def test_assert_machine_input_bad_dtbin(self):
        machine = mch.Machine(**MACHINE_ARGS)
        
        machine.dtbin = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='dtbin with value None should raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.dtbin = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='dtbin with value of 0 should raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_dturns(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.dturns = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='dturns with value None should raise an Exception'):
            asrt.assert_machine_input(machine)
        
        machine.dturns = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='dturns with value of 0 should raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_filmsss(self):
        machine = mch.Machine(**MACHINE_ARGS)

        # Testing filmstart
        machine.filmstart = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='filmstart with value None should raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.filmstart = -1
        with self.assertRaises(
                expt.MachineParameterError,
                msg='filmstart with negative value should raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.filmstart = 2
        with self.assertRaises(
                expt.MachineParameterError,
                msg='filmstart with value larger than filmstop '
                    'should raise an Exception'):
            asrt.assert_machine_input(machine)
        machine.filmstart = 0

        # Testing filmstep

        machine.filmstep = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='filmstep with value None should raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.filmstep = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='filmstep with negative value should raise an Exception'):
            asrt.assert_machine_input(machine)
        machine.filmstep = 1

        # Testing filmstop

        machine.filmstop = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='filmstop with value None should raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_niter(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.niter = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='niter with value None should raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.niter = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='niter with value less than 1 should raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_snpt(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.snpt = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='snpt with value None should raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.snpt = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='snpt with value less than 1 should raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_ref_frames(self):
        machine = mch.Machine(**MACHINE_ARGS)

        # Testing machine reference frame
        machine.machine_ref_frame = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='machine_ref_frame with value None should '
                    'raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.machine_ref_frame = -1
        with self.assertRaises(
                expt.MachineParameterError,
                msg='machine_ref_frame with value less than zero should '
                    'raise an Exception'):
            asrt.assert_machine_input(machine)
        machine.machine_ref_frame = 1

        # Testing beam reference frame
        machine.beam_ref_frame = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='beam_ref_frame with value None should '
                    'raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.beam_ref_frame = -1
        with self.assertRaises(
                expt.MachineParameterError,
                msg='beam_ref_frame with value less than zero should '
                    'raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_hnum(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.h_num = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='h_num with value None should raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.h_num = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='h_num with value less than one should '
                    'raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_h_ratio(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.h_ratio = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='h_ratio with value None should raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.h_ratio = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='h_ratio with value less than one should '
                    'raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_b0(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.b0 = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='b0 with value None should raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.b0 = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='b0 with with value less than or equal to zero should '
                    'raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_mean_orbit_rad(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.mean_orbit_rad = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='mean_orbit_rad with value None should '
                    'raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.mean_orbit_rad = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='mean_orbit_rad with with value less than or '
                    'equal to zero should raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_bending_rad(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.bending_rad = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='bending_rad with value None should '
                    'raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.bending_rad = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='bending_rad with with value less than or '
                    'equal to zero should raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_e_rest(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.e_rest = None
        with self.assertRaises(
                expt.MachineParameterError,
                msg='e_rest with value None should '
                    'raise an Exception'):
            asrt.assert_machine_input(machine)

        machine.e_rest = 0
        with self.assertRaises(
                expt.MachineParameterError,
                msg='e_rest with with value less than or '
                    'equal to zero should raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_pickup_sensitivity(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.pickup_sensitivity = -1
        with self.assertRaises(
                expt.SpaceChargeParameterError,
                msg='pickup_sensitivity with with value less than '
                    'zero should raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_machine_input_bad_g_coupling(self):
        machine = mch.Machine(**MACHINE_ARGS)

        machine.g_coupling = -1
        with self.assertRaises(
                expt.SpaceChargeParameterError,
                msg='g_coupling with with value less than '
                    'zero should raise an Exception'):
            asrt.assert_machine_input(machine)

    def test_assert_frame_input(self):
        frame = tomoin.Frames(
                    framecount=10, framelength=10, skip_frames=2,
                    skip_bins_start=2, skip_bins_end=2, rebin=1, dtbin=1e-7)

        # Checking nframes
        frame.nframes = 0
        with self.assertRaises(
                expt.InputError,
                msg='nframes equal to zero should raise an Exception'):
            asrt.assert_frame_inputs(frame)
        frame.nframes = 10

        # Checking skip_frames
        frame.skip_frames = -1
        with self.assertRaises(
                expt.InputError,
                msg='skip_frames with negative value '
                    'should raise an Exception'):
            asrt.assert_frame_inputs(frame)
        
        frame.skip_frames = frame.nframes 
        with self.assertRaises(
                expt.InputError,
                msg='skip_frames with value equal to nframes '
                    'should raise an Exception'):
            asrt.assert_frame_inputs(frame)
        frame.skip_frames = 2

        # Checking nbins_frame
        frame.nbins_frame = -1
        with self.assertRaises(
                expt.InputError,
                msg='nbins_frame with value lower than zero '
                    'should raise an Exception'):
            asrt.assert_frame_inputs(frame)
        frame.nbins_frame = 10

        # Checking skip_bins_start
        frame.skip_bins_start = -1
        with self.assertRaises(
                expt.InputError,
                msg='skip_bins_start with value lower than zero '
                    'should raise an Exception'):
            asrt.assert_frame_inputs(frame)

        frame.skip_bins_start = frame.nbins_frame
        with self.assertRaises(
                expt.InputError,
                msg='skip_bins_start with value equal to nbins_frame '
                    'should raise an Exception'):
            asrt.assert_frame_inputs(frame)
        frame.skip_bins_start = 2

        # Checking skip_bins_end
        frame.skip_bins_end = -1
        with self.assertRaises(
                expt.InputError,
                msg='skip_bins_end with value lower than zero '
                    'should raise an Exception'):
            asrt.assert_frame_inputs(frame)

        frame.skip_bins_end = frame.nbins_frame
        with self.assertRaises(
                expt.InputError,
                msg='skip_bins_end with value equal to nbins_frame '
                    'should raise an Exception'):
            asrt.assert_frame_inputs(frame)
        frame.skip_bins_end = 2

        # Checking total skipped bins
        frame.skip_bins_start = 5
        frame.skip_bins_end = 5
        with self.assertRaises(
                expt.InputError,
                msg='total number of skipped bins greater than or equal to '
                    'nbins_frame should raise an Exception'):
            asrt.assert_frame_inputs(frame)

        frame.skip_bins_start = 2
        frame.skip_bins_end = 2

        frame.rebin = 0
        with self.assertRaises(
                expt.InputError,
                msg='skip_bins_start with value lower than zero '
                    'should raise an Exception'):
            asrt.assert_frame_inputs(frame)

    def test_assert_var_not_none(self):
        error_class = AssertionError
        with self.assertRaises(error_class,
                               msg='An exception should have been raised'):
            asrt.assert_var_not_none(None, 'test', error_class)
