"""Unit-tests for the tomo_input module.

Run as python test_tomo_input.py in console or via coverage
"""

import os
import unittest

import numpy as np
import numpy.testing as nptest

import tomo.tracking.machine as mch
from tomo import exceptions as expt
import tomo.utils.tomo_input as tomoin

# All values retrieved from INDIVShavingC325.dat
frame_input_args = {
    'raw_data_path':       '',
    'framecount':          150,
    'skip_frames':         0,
    'framelength':         1000,
    'dtbin':               9.999999999999999E-10,
    'skip_bins_start':     170,
    'skip_bins_end':       70,
    'rebin':               3 
}

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


class TestTomoIn(unittest.TestCase):

    # Tests for frame class
    # ---------------------

    def test_frames_set_raw_data_fails(self):
        frames = tomoin.Frames(**frame_input_args)

        with self.assertRaises(expt.RawDataImportError,
                               msg='raw data provided as scalar should raise '
                                   'an exception'):
            frames.raw_data = 1

        with self.assertRaises(expt.RawDataImportError,
                               msg='Wrong amount of data points should raise '
                                   'an exception'):
            frames.raw_data = [1]

    def test_nprofs_correct(self):
        frames = tomoin.Frames(**frame_input_args)
        nprof = frames.nprofs()
        self.assertEqual(nprof, 150, msg='Error in calculation of nprofs')

    def test_nbins_correct(self):
        frames = tomoin.Frames(**frame_input_args)
        nbins = frames.nbins()
        self.assertEqual(nbins, 760, msg='Error in calculation of nbins')

    def test_to_waterfall_skip_bins_both_sides(self):
        nframes = 3
        nbins_frame = 30
        dtbin = 9.999999999999999E-10
        skip_frames = 1
        skip_bins_start = 2
        skip_bins_end = 2
        rebin = 1

        ndata = nframes * nbins_frame

        raw_data = np.arange(ndata)

        frames = tomoin.Frames(
                    nframes, nbins_frame, skip_frames, skip_bins_start,
                    skip_bins_end, rebin, dtbin)

        waterfall = frames.to_waterfall(raw_data)

        correct = np.array([[32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                             56, 57],
                            [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
                             74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                             86, 87]])
        nptest.assert_equal(
            waterfall, correct,
            err_msg='error in conversion from raw data to waterfall')

    def test_to_waterfall_skip_start_bins(self):
        nframes = 3
        nbins_frame = 30
        dtbin = 9.999999999999999E-10
        skip_frames = 1
        skip_bins_start = 6
        skip_bins_end = 0
        rebin = 1

        ndata = nframes * nbins_frame

        raw_data = np.arange(ndata)

        frames = tomoin.Frames(
                    nframes, nbins_frame, skip_frames, skip_bins_start,
                    skip_bins_end, rebin, dtbin)

        waterfall = frames.to_waterfall(raw_data)

        correct = np.array([[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                             48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                            [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                             78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]])

        nptest.assert_equal(
            waterfall, correct,
            err_msg='error in conversion from raw data to waterfall')

    def test_to_waterfall_wrong_number_raw_data(self):
        nframes = 3
        nbins_frame = 30
        dtbin = 9.999999999999999E-10
        skip_frames = 1
        skip_bins_start = 6
        skip_bins_end = 0
        rebin = 1

        frames = tomoin.Frames(
                    nframes, nbins_frame, skip_frames, skip_bins_start,
                    skip_bins_end, rebin, dtbin)

        raw_data = np.arange(20)

        with self.assertRaises(expt.RawDataImportError,
                               msg='Wrong size of raw data relative to the '
                                   'given parameters should '
                                   'raise an exception'):
            frames.to_waterfall(raw_data)

    def test_to_waterfall_raw_data_no_iter(self):
        nframes = 3
        nbins_frame = 30
        dtbin = 9.999999999999999E-10
        skip_frames = 1
        skip_bins_start = 6
        skip_bins_end = 0
        rebin = 1

        frames = tomoin.Frames(
                    nframes, nbins_frame, skip_frames, skip_bins_start,
                    skip_bins_end, rebin, dtbin)

        raw_data = 1
        with self.assertRaises(expt.RawDataImportError,
                               msg='Raw data as scalar should raise '
                                   'an exception'):
            frames.to_waterfall(raw_data)

    # Tests for module functions
    # --------------------------

    def test_raw_data_to_profiles_no_iter_fails(self):
        machine = mch.Machine(**MACHINE_ARGS)
        waterfall = 1
        rbn = 1
        sampling_time = machine.dtbin

        with self.assertRaises(expt.WaterfallError,
                               msg='Waterfall as non iterable should raise '
                                   'an exception'):
            tomoin.raw_data_to_profiles(waterfall, machine, rbn, sampling_time)

    def test_raw_data_to_profiles_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        raw_data = self._load_raw_data()

        waterfall = raw_data.reshape((150, 1000))
        waterfall = waterfall[:, 170: -70]
        rbn = 3
        s_time = machine.dtbin

        profiles = tomoin.raw_data_to_profiles(
                            waterfall, machine, rbn, s_time)

        correct_waterfall = self._load_waterfall()

        # Checking that waterfall was rebinned correctly
        nptest.assert_almost_equal(
            profiles.waterfall, correct_waterfall,
            err_msg='An found in the profiles.waterfall, after converting'
                    'from waterfall of raw data to profiles')

        # Checking if machine.dtbin was updated correctly
        correct_dtbin_e9 = 2.9999999999999996

        self.assertAlmostEqual(
            machine.dtbin * 1e9, correct_dtbin_e9,
            msg='The value of machine.dtbin not correct after rebinning.')

        # Checking that machine.synch_part_x was updated correctly
        correct_synch_part_x = 111.33333333333336
        self.assertAlmostEqual(
            machine.synch_part_x, correct_synch_part_x,
            msg='The value of machine.synch_part_x not '
                'correct after rebinning.')

        # Checking that the sampling time set in the profiles object
        # is correct.
        self.assertEqual(
            profiles.sampling_time, s_time,
            msg='The profile objects sampling time should have '
                'the value of the original dtbin')

    def test_txt_input_to_machine_machine_correct(self):
        data_path = self._find_resource_dir()
        infile = os.path.join(data_path, 'parameters_INDIVShavingC325.dat')
        
        with open(infile, 'r') as f: 
            file_lines = f.readlines()
        machine, _ = tomoin.txt_input_to_machine(file_lines)

        self.assertEqual(machine.demax, MACHINE_ARGS['demax'],
                         msg='demax set incorrectly')

        self.assertEqual(machine.dturns, MACHINE_ARGS['dturns'],
                         msg='dturns set incorrectly')

        self.assertEqual(machine.vrf1, MACHINE_ARGS['vrf1'],
                         msg='vrf1 set incorrectly')

        self.assertEqual(machine.vrf1dot, MACHINE_ARGS['vrf1dot'],
                         msg='vrf1dot set incorrectly')

        self.assertEqual(machine.vrf2, MACHINE_ARGS['vrf2'],
                         msg='vrf2 set incorrectly')

        self.assertEqual(machine.vrf2dot, MACHINE_ARGS['vrf2dot'],
                         msg='vrf2dot set incorrectly')

        self.assertEqual(machine.mean_orbit_rad,
                         MACHINE_ARGS['mean_orbit_rad'],
                         msg='mean_orbit_rad set incorrectly')

        self.assertEqual(machine.bending_rad, MACHINE_ARGS['bending_rad'],
                         msg='bending_rad set incorrectly')

        self.assertEqual(machine.b0, MACHINE_ARGS['b0'],
                         msg='b0 set incorrectly')

        self.assertEqual(machine.bdot, MACHINE_ARGS['bdot'],
                         msg='bdot set incorrectly')

        self.assertEqual(machine.phi12, MACHINE_ARGS['phi12'],
                         msg='phi12 set incorrectly')

        self.assertEqual(machine.h_ratio, MACHINE_ARGS['h_ratio'],
                         msg='h_ratio set incorrectly')

        self.assertEqual(machine.trans_gamma, MACHINE_ARGS['trans_gamma'],
                         msg='trans_gamma set incorrectly')

        self.assertEqual(machine.e_rest, MACHINE_ARGS['rest_energy'],
                         msg='rest_energy set incorrectly')

        self.assertEqual(machine.q, MACHINE_ARGS['charge'],
                         msg='q set incorrectly')

        self.assertEqual(machine.g_coupling, MACHINE_ARGS['g_coupling'],
                         msg='g_coupling set incorrectly')

        self.assertEqual(machine.zwall_over_n, MACHINE_ARGS['zwall_over_n'],
                         msg='zwall_over_n set incorrectly')

        self.assertEqual(machine.min_dt, MACHINE_ARGS['min_dt'],
                         msg='min_dt set incorrectly')

        self.assertEqual(machine.max_dt, MACHINE_ARGS['max_dt'],
                         msg='max_dt set incorrectly')

        self.assertEqual(machine.pickup_sensitivity,
                         MACHINE_ARGS['pickup_sensitivity'],
                         msg='pickup_sensitivity set incorrectly')

        self.assertEqual(machine.synch_part_x, MACHINE_ARGS['synch_part_x'],
                         msg='synch_part_x set incorrectly')

        self.assertEqual(machine.dtbin, MACHINE_ARGS['dtbin'],
                         msg='dtbin set incorrectly')

        self.assertEqual(machine.self_field_flag,
                         MACHINE_ARGS['self_field_flag'],
                         msg='self_field_flag set incorrectly')

        self.assertEqual(machine.full_pp_flag,
                         MACHINE_ARGS['full_pp_flag'],
                         msg='full_pp_flag set incorrectly')

        self.assertEqual(machine.machine_ref_frame,
                         MACHINE_ARGS['machine_ref_frame'],
                         msg='machine_ref_frame set incorrectly')

        self.assertEqual(machine.beam_ref_frame,
                         MACHINE_ARGS['beam_ref_frame'],
                         msg='beam_ref_frame set incorrectly')

        self.assertEqual(machine.snpt, MACHINE_ARGS['snpt'],
                         msg='snpt set incorrectly')
        self.assertEqual(machine.niter, MACHINE_ARGS['niter'],
                         msg='niter set incorrectly')
        self.assertEqual(machine.filmstart, MACHINE_ARGS['filmstart'],
                         msg='filmstart set incorrectly')
        self.assertEqual(machine.filmstop, MACHINE_ARGS['filmstop'],
                         msg='filmstop set incorrectly')
        self.assertEqual(machine.filmstep, MACHINE_ARGS['filmstep'],
                         msg='filmstep set incorrectly')
        
        nprofiles = 150
        self.assertEqual(machine.nprofiles, nprofiles,
                         msg='nprofiles set incorrectly')
        nbins = 760
        self.assertEqual(machine.nbins, nbins,
                         msg='nprofiles set incorrectly')

    def test_txt_input_to_machine_frames_correct(self):
        data_path = self._find_resource_dir()
        infile = os.path.join(data_path, 'parameters_INDIVShavingC325.dat')
        
        with open(infile, 'r') as f: 
            file_lines = f.readlines()
        _, frames = tomoin.txt_input_to_machine(file_lines)

        self.assertEqual(frames.nframes, frame_input_args['framecount'],
                         msg='The number of frames is not set correctly')

        self.assertEqual(frames.nbins_frame, frame_input_args['framelength'],
                         msg='The number of frame bins is not set correctly')

        self.assertEqual(frames.skip_frames, frame_input_args['skip_frames'],
                         msg='The number of frames to be skipped '
                             'is not set correctly')

        self.assertEqual(frames.skip_bins_start,
                         frame_input_args['skip_bins_start'],
                         msg='The number binst to be skipped in the start '
                             'of each frame is not set correctly')

        self.assertEqual(frames.skip_bins_end,
                         frame_input_args['skip_bins_end'],
                         msg='The number binst to be skipped in the end '
                             'of each frame is not set correctly')

        self.assertEqual(frames.rebin,
                         frame_input_args['rebin'],
                         msg='The rebin factor is set incorrectly')

        self.assertEqual(frames.sampling_time,
                         frame_input_args['dtbin'],
                         msg='The sampling time factor is set incorrectly')

    def test_txt_input_to_machine_no_iter_fails(self):
        not_iterable = 1

        with self.assertRaises(expt.InputError,
                               msg='non iterable input should '
                                   'raise an exception'):
            _, _ = tomoin.txt_input_to_machine(not_iterable)

    def test_txt_input_to_machine_wrong_len_fails(self):
        wrong_length_iter = [1, 2, 3]

        with self.assertRaises(expt.InputError,
                               msg='non iterable of the wrong length '
                                   'should raise an exception'):
            _, _ = tomoin.txt_input_to_machine(wrong_length_iter)

    def _load_waterfall(self):
        data_path = self._find_resource_dir()
        waterfall = np.load(os.path.join(
                        data_path, 'waterfall_INDIVShavingC325.npy'))
        return waterfall

    def _load_raw_data(self):
        data_path = self._find_resource_dir()
        raw_data = np.genfromtxt(
                    os.path.join(data_path,
                                'raw_data_INDIVShavingC325.dat'))
        return raw_data

    def _find_resource_dir(self):
        base_dir = os.path.split(os.path.realpath(__file__))[0]
        base_dir = os.path.split(base_dir)[0]
        data_path = os.path.join(base_dir, 'resources')
        return data_path
