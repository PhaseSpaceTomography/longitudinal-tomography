"""Unit-tests for the tomo_input module.

Run as python test_tomo_input.py in console or via coverage
"""

import os
import tempfile
import unittest

import numpy as np
import numpy.testing as nptest
import pydantic

from .. import commons
import longitudinal_tomography.tracking.machine as mch
import longitudinal_tomography.tracking.machine_base as machine_base
from longitudinal_tomography import exceptions as expt
import longitudinal_tomography.utils.tomo_input as tomoin

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

MACHINE_ARGS = commons.get_machine_args()

# Machine args that are derived from the frame parameters rather than read
# straight out of the .dat file, so have no Parameters field of their own.
DERIVED_MACHINE_ARGS = ('nprofiles', 'nbins', 'min_dt', 'max_dt')


def _resource_dir() -> str:
    base_dir = os.path.split(os.path.realpath(__file__))[0]
    base_dir = os.path.split(base_dir)[0]
    return os.path.join(base_dir, 'resources')


def _parameter_lines() -> list:
    """The 98 header lines of the reference .dat file, without raw data."""
    infile = os.path.join(_resource_dir(), 'parameters_INDIVShavingC325.dat')
    with open(infile, 'r') as f:
        return f.readlines()


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
        nprof = frames.nprofs
        self.assertEqual(nprof, 150, msg='Error in calculation of nprofs')

    def test_nbins_correct(self):
        frames = tomoin.Frames(**frame_input_args)
        nbins = frames.nbins
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


# Minimal set of fields with no default, for building a Parameters by hand.
PARAMETER_ARGS = {
    'framecount': 150,
    'skip_frames': 0,
    'framelength': 1000,
    'dtbin': 9.999999999999999E-10,
    'dturns': 5,
    'skip_bins_start': 170,
    'skip_bins_end': 70,
    'min_dt_bin': 0,
    'max_dt_bin': 0,
    'rebin': 3,
    'filmstep': 1,
    'vrf1': 2637.197030932989,
    'b0': 0.15722,
    'mean_orbit_rad': 25.0,
    'bending_rad': 8.239,
    'trans_gamma': 4.1,
    'rest_energy': 0.93827231E9,
}


class TestParameters(unittest.TestCase):

    def test_from_dat_lines_matches_machine_args(self):
        params = tomoin.Parameters.from_dat_lines(_parameter_lines())

        for name, expected in MACHINE_ARGS.items():
            if name in DERIVED_MACHINE_ARGS:
                continue
            self.assertEqual(
                getattr(params, name), expected,
                msg=f'{name} parsed incorrectly from the .dat file')

    def test_from_dat_lines_frame_fields(self):
        params = tomoin.Parameters.from_dat_lines(_parameter_lines())

        for name in ('framecount', 'skip_frames', 'framelength',
                     'skip_bins_start', 'skip_bins_end', 'rebin'):
            self.assertEqual(
                getattr(params, name), frame_input_args[name],
                msg=f'{name} parsed incorrectly from the .dat file')

        self.assertEqual(params.min_dt_bin, 0,
                         msg='min_dt_bin parsed incorrectly')
        self.assertEqual(params.max_dt_bin, 0,
                         msg='max_dt_bin parsed incorrectly')

    def test_from_dat_lines_converts_fortran_indices(self):
        lines = _parameter_lines()
        for name, value in (('filmstart', 3), ('filmstop', 5),
                            ('beam_ref_frame', 7), ('machine_ref_frame', 9)):
            lines[tomoin.DAT_FILE_LINE_INDEX[name]] = f'{value}\n'
        # dturns is an ordinary int field, it must survive untouched.
        lines[tomoin.DAT_FILE_LINE_INDEX['dturns']] = '9\n'

        params = tomoin.Parameters.from_dat_lines(lines)

        self.assertEqual(params.filmstart, 2,
                         msg='filmstart not converted to 0-based indexing')
        self.assertEqual(params.filmstop, 4,
                         msg='filmstop not converted to 0-based indexing')
        self.assertEqual(
            params.beam_ref_frame, 6,
            msg='beam_ref_frame not converted to 0-based indexing')
        self.assertEqual(
            params.machine_ref_frame, 8,
            msg='machine_ref_frame not converted to 0-based indexing')
        self.assertEqual(params.dturns, 9,
                         msg='dturns should not be index-converted')

    def test_from_dat_lines_leaves_input_unchanged(self):
        lines = _parameter_lines()
        original = list(lines)

        tomoin.Parameters.from_dat_lines(lines)

        self.assertEqual(lines, original,
                         msg='from_dat_lines must not mutate its input')

    def test_from_dat_lines_strips_line_endings(self):
        lines = [line.rstrip('\n') + '\r\n' for line in _parameter_lines()]

        params = tomoin.Parameters.from_dat_lines(lines)

        self.assertEqual(params.output_dir, MACHINE_ARGS['output_dir'],
                         msg='carriage returns not stripped from .dat lines')

    def test_from_dat_lines_strips_padding(self):
        # Fortran writers pad numeric fields; bool coercion of a padded
        # value fails unless the whole line is stripped.
        lines = [f'   {line.strip()}  \n' for line in _parameter_lines()]

        params = tomoin.Parameters.from_dat_lines(lines)
        expected = tomoin.Parameters.from_dat_lines(_parameter_lines())

        self.assertEqual(params.model_dump(), expected.model_dump(),
                         msg='padded .dat lines should parse identically to '
                             'unpadded ones')

    def test_from_dat_lines_unparsable_index_fails(self):
        # The index fields are converted before pydantic sees them, so they
        # need their own error handling.
        for name in tomoin.FORTRAN_TO_C_FIELDS:
            with self.subTest(field=name):
                lines = _parameter_lines()
                lines[tomoin.DAT_FILE_LINE_INDEX[name]] = 'not a number\n'

                with self.assertRaises(
                        expt.InputError,
                        msg=f'an uncoercible {name} should raise '
                            f'an InputError'):
                    tomoin.Parameters.from_dat_lines(lines)

    def test_defaults_match_machine_defaults(self):
        defaults = {**machine_base._machine_base_defaults,
                    **mch._machine_opts_def}

        for name, expected in defaults.items():
            if name not in tomoin.Parameters.model_fields:
                continue
            with self.subTest(field=name):
                self.assertEqual(
                    tomoin.Parameters.model_fields[name].default, expected,
                    msg=f'the default for {name} differs from the one used '
                        f'by Machine')

    def test_from_dat_lines_no_iter_fails(self):
        with self.assertRaises(expt.InputError,
                               msg='non iterable input should '
                                   'raise an exception'):
            tomoin.Parameters.from_dat_lines(1)

    def test_from_dat_lines_wrong_len_fails(self):
        with self.assertRaises(expt.InputError,
                               msg='input of the wrong length should '
                                   'raise an exception'):
            tomoin.Parameters.from_dat_lines(['1', '2', '3'])

    def test_from_dat_lines_invalid_value_fails(self):
        lines = _parameter_lines()
        lines[tomoin.DAT_FILE_LINE_INDEX['rebin']] = '0\n'

        with self.assertRaises(expt.InputError,
                               msg='a value violating a field constraint '
                                   'should raise an InputError'):
            tomoin.Parameters.from_dat_lines(lines)

    def test_from_dat_lines_unparsable_value_fails(self):
        lines = _parameter_lines()
        lines[tomoin.DAT_FILE_LINE_INDEX['dturns']] = 'not a number\n'

        with self.assertRaises(expt.InputError,
                               msg='a value that cannot be coerced should '
                                   'raise an InputError'):
            tomoin.Parameters.from_dat_lines(lines)

    def test_rejects_out_of_range_values(self):
        for name, value in (('framecount', 0), ('framelength', 0),
                            ('skip_frames', -1), ('skip_bins_start', -1),
                            ('skip_bins_end', -1), ('rebin', 0),
                            ('dtbin', 0.0), ('dturns', 0), ('filmstep', 0),
                            ('niter', 0), ('snpt', 0), ('h_num', 0),
                            ('h_ratio', 0), ('b0', 0.0), ('charge', 0),
                            ('mean_orbit_rad', 0.0), ('bending_rad', 0.0),
                            ('rest_energy', 0.0), ('min_dt_bin', -1),
                            ('max_dt_bin', -1)):
            with self.subTest(field=name):
                with self.assertRaises(
                        pydantic.ValidationError,
                        msg=f'{name}={value} should be rejected'):
                    tomoin.Parameters(**{**PARAMETER_ARGS, name: value})

    def test_rejects_cross_field_violations(self):
        for name, value in (('skip_frames', 150), ('skip_bins_start', 1000),
                            ('skip_bins_end', 1000), ('synch_part_x', 1000.0),
                            ('filmstart', 150), ('filmstop', 150),
                            ('beam_ref_frame', 150),
                            ('machine_ref_frame', 150)):
            with self.subTest(field=name):
                with self.assertRaises(
                        pydantic.ValidationError,
                        msg=f'{name}={value} exceeds its frame bound and '
                            f'should be rejected'):
                    tomoin.Parameters(**{**PARAMETER_ARGS, name: value})

    def test_accepts_numpy_scalars(self):
        params = tomoin.Parameters(
            **{**PARAMETER_ARGS, 'framecount': np.int32(150),
               'dtbin': np.float32(1e-9), 'h_num': np.int64(1)})

        self.assertIsInstance(params.framecount, int,
                              msg='numpy ints should be coerced to int')
        self.assertIsInstance(params.dtbin, float,
                              msg='numpy floats should be coerced to float')
        self.assertEqual(params.h_num, 1.0,
                         msg='numpy int should be coerced to a float field')

    def test_rejects_fractional_float_for_int_field(self):
        with self.assertRaises(pydantic.ValidationError,
                               msg='a fractional float should not be '
                                   'silently truncated to an int field'):
            tomoin.Parameters(**{**PARAMETER_ARGS, 'dturns': 5.5})

    def test_flags_parsed_from_dat_digits(self):
        for value, expected in (('0', False), ('1', True)):
            with self.subTest(value=value):
                lines = _parameter_lines()
                lines[tomoin.DAT_FILE_LINE_INDEX['full_pp_flag']] = \
                    f'{value}\n'
                lines[tomoin.DAT_FILE_LINE_INDEX['self_field_flag']] = \
                    f'{value}\n'

                params = tomoin.Parameters.from_dat_lines(lines)

                self.assertIs(params.full_pp_flag, expected,
                              msg='full_pp_flag parsed incorrectly')
                self.assertIs(params.self_field_flag, expected,
                              msg='self_field_flag parsed incorrectly')


class TestFramesValidation(unittest.TestCase):

    def test_rejects_out_of_range_values(self):
        for name, value in (('framecount', 0), ('framelength', 0),
                            ('skip_frames', -1), ('skip_bins_start', -1),
                            ('skip_bins_end', -1), ('rebin', 0),
                            ('dtbin', 0.0)):
            with self.subTest(field=name):
                with self.assertRaises(
                        pydantic.ValidationError,
                        msg=f'{name}={value} should be rejected'):
                    tomoin.Frames(**{**frame_input_args, name: value})

    def test_rejects_cross_field_violations(self):
        for name, value in (('skip_frames', 150), ('skip_bins_start', 1000),
                            ('skip_bins_end', 1000)):
            with self.subTest(field=name):
                with self.assertRaises(
                        pydantic.ValidationError,
                        msg=f'{name}={value} exceeds its frame bound and '
                            f'should be rejected'):
                    tomoin.Frames(**{**frame_input_args, name: value})

    def test_attributes_remain_mutable(self):
        # assert_frame_inputs is the use-time check, so construction-time
        # validation must not lock the object down.
        frames = tomoin.Frames(**frame_input_args)
        frames.skip_frames = -1

        self.assertEqual(frames.skip_frames, -1,
                         msg='Frames attributes should not be revalidated '
                             'on assignment')

    def test_from_parameters_maps_fields(self):
        params = tomoin.Parameters.from_dat_lines(_parameter_lines())
        frames = tomoin.Frames.from_parameters(params, raw_data_path='a/path')

        self.assertEqual(frames.nframes, params.framecount,
                         msg='nframes not taken from framecount')
        self.assertEqual(frames.nbins_frame, params.framelength,
                         msg='nbins_frame not taken from framelength')
        self.assertEqual(frames.skip_frames, params.skip_frames,
                         msg='skip_frames not taken from parameters')
        self.assertEqual(frames.skip_bins_start, params.skip_bins_start,
                         msg='skip_bins_start not taken from parameters')
        self.assertEqual(frames.skip_bins_end, params.skip_bins_end,
                         msg='skip_bins_end not taken from parameters')
        self.assertEqual(frames.rebin, params.rebin,
                         msg='rebin not taken from parameters')
        self.assertEqual(frames.sampling_time, params.dtbin,
                         msg='sampling_time not taken from dtbin')
        self.assertEqual(frames.raw_data_path, 'a/path',
                         msg='raw_data_path not set')

    def test_from_parameters_default_raw_data_path(self):
        params = tomoin.Parameters.from_dat_lines(_parameter_lines())
        frames = tomoin.Frames.from_parameters(params)

        self.assertEqual(frames.raw_data_path, '',
                         msg='raw_data_path should default to an empty string')


class TestFramesConstruction(unittest.TestCase):
    """The legacy constructor names differ from the field names, so every
    way of building a Frames must give the same object.
    """

    def _field_args(self) -> dict:
        return {
            'nframes': frame_input_args['framecount'],
            'nbins_frame': frame_input_args['framelength'],
            'skip_frames': frame_input_args['skip_frames'],
            'skip_bins_start': frame_input_args['skip_bins_start'],
            'skip_bins_end': frame_input_args['skip_bins_end'],
            'rebin': frame_input_args['rebin'],
            'sampling_time': frame_input_args['dtbin'],
            'raw_data_path': frame_input_args['raw_data_path'],
        }

    def test_positional_matches_keyword(self):
        positional = tomoin.Frames(
            frame_input_args['framecount'], frame_input_args['framelength'],
            frame_input_args['skip_frames'],
            frame_input_args['skip_bins_start'],
            frame_input_args['skip_bins_end'], frame_input_args['rebin'],
            frame_input_args['dtbin'])

        self.assertEqual(positional, tomoin.Frames(**frame_input_args),
                         msg='positional and legacy keyword construction '
                             'should agree')

    def test_field_names_accepted(self):
        self.assertEqual(tomoin.Frames(**self._field_args()),
                         tomoin.Frames(**frame_input_args),
                         msg='construction by field name and by legacy name '
                             'should agree')

    def test_too_many_positional_args_fails(self):
        with self.assertRaises(TypeError,
                               msg='more positional arguments than the '
                                   'legacy signature takes should fail'):
            tomoin.Frames(150, 1000, 0, 170, 70, 3, 1e-9, '', 'extra')

    def test_duplicate_argument_fails(self):
        with self.assertRaises(TypeError,
                               msg='the same argument given positionally and '
                                   'by keyword should fail'):
            tomoin.Frames(150, 1000, 0, 170, 70, 3, 1e-9, framecount=150)

    def test_model_validate_round_trip(self):
        frames = tomoin.Frames(**frame_input_args)

        self.assertEqual(tomoin.Frames.model_validate(frames.model_dump()),
                         frames,
                         msg='a Frames should survive a dump/validate cycle')

    def test_equality_compares_raw_data(self):
        a = tomoin.Frames(**frame_input_args)
        b = tomoin.Frames(**frame_input_args)
        self.assertEqual(a, b, msg='identical frames should compare equal')

        a.raw_data = np.arange(150 * 1000)
        self.assertNotEqual(
            a, b, msg='frames should differ when only one holds raw data')

        b.raw_data = np.arange(150 * 1000)
        self.assertEqual(a, b,
                         msg='frames holding equal raw data should compare '
                             'equal rather than raise')

        b.raw_data = np.ones(150 * 1000)
        self.assertNotEqual(a, b,
                            msg='frames holding different raw data should '
                                'not compare equal')

    def test_equality_with_other_type(self):
        self.assertNotEqual(tomoin.Frames(**frame_input_args), 1,
                            msg='a Frames should not equal a non-Frames')


class TestTomographyData(unittest.TestCase):

    def _dat_path(self) -> str:
        return os.path.join(_resource_dir(), 'INDIVShavingC325.dat')

    def test_from_dat_file_parameters(self):
        data = tomoin.TomographyData.from_dat_file(self._dat_path())

        self.assertIsInstance(data.parameters, tomoin.Parameters,
                              msg='parameters should be a Parameters')
        self.assertIsInstance(data.frames, tomoin.Frames,
                              msg='frames should be a Frames')

        expected = tomoin.Parameters.from_dat_lines(_parameter_lines())
        self.assertEqual(
            data.parameters.model_dump(), expected.model_dump(),
            msg='parameters read from the file differ from the same file '
                'parsed as lines')

    def test_from_dat_file_frames(self):
        data = tomoin.TomographyData.from_dat_file(self._dat_path())

        self.assertEqual(data.frames.nprofs, 150,
                         msg='nprofs not derived from the parsed parameters')
        self.assertEqual(data.frames.nbins, 760,
                         msg='nbins not derived from the parsed parameters')
        self.assertEqual(data.frames.raw_data_path, 'pipe',
                         msg='raw_data_path not taken from the .dat file')

    def test_from_dat_file_holds_raw_data(self):
        data = tomoin.TomographyData.from_dat_file(self._dat_path())

        self.assertIsNotNone(data.frames.raw_data,
                             msg='raw data should be attached to the frames')
        self.assertEqual(len(data.frames.raw_data), 150 * 1000,
                         msg='wrong amount of raw data attached')

    def test_from_dat_file_waterfall_correct(self):
        data = tomoin.TomographyData.from_dat_file(self._dat_path())

        waterfall = data.frames.to_waterfall(data.frames.raw_data)

        machine = mch.Machine(**MACHINE_ARGS)
        profiles = tomoin.raw_data_to_profiles(
            waterfall, machine, data.frames.rebin, data.frames.sampling_time)

        nptest.assert_almost_equal(
            profiles.waterfall, commons.load_waterfall(),
            err_msg='waterfall built from TomographyData does not match the '
                    'reference waterfall')

    def test_from_dat_file_agrees_with_txt_input_to_machine(self):
        raw_params, _ = tomoin.get_user_input(self._dat_path())
        machine, frames = tomoin.txt_input_to_machine(raw_params)

        data = tomoin.TomographyData.from_dat_file(self._dat_path())

        for name in MACHINE_ARGS:
            if name in DERIVED_MACHINE_ARGS:
                continue
            self.assertEqual(
                getattr(data.parameters, name),
                getattr(machine, {'rest_energy': 'e_rest',
                                  'charge': 'q'}.get(name, name)),
                msg=f'{name} differs between TomographyData and '
                    f'txt_input_to_machine')

        self.assertEqual(data.frames.nprofs, machine.nprofiles,
                         msg='nprofs differs from machine.nprofiles')
        self.assertEqual(data.frames.nbins, machine.nbins,
                         msg='nbins differs from machine.nbins')
        self.assertEqual(data.frames.model_dump(), frames.model_dump(),
                         msg='frames differ between TomographyData and '
                             'txt_input_to_machine')

    def test_json_round_trip(self):
        data = tomoin.TomographyData.from_dat_file(self._dat_path())

        restored = tomoin.TomographyData.model_validate_json(
            data.model_dump_json())

        self.assertEqual(restored.parameters, data.parameters,
                         msg='parameters should survive a JSON round trip')
        self.assertEqual(restored.frames.model_dump(),
                         data.frames.model_dump(),
                         msg='frames should survive a JSON round trip')
        self.assertIsNone(restored.frames.raw_data,
                          msg='raw data is not serialised, so it should not '
                              'silently reappear after a round trip')

    def test_from_dat_file_missing_file_fails(self):
        with self.assertRaises(expt.InputError,
                               msg='a missing input file should raise '
                                   'an exception'):
            tomoin.TomographyData.from_dat_file('does/not/exist.dat')


class TestDatFileWriting(unittest.TestCase):

    def _dat_path(self) -> str:
        return os.path.join(_resource_dir(), 'INDIVShavingC325.dat')

    def _round_trip(self) -> tuple:
        original = tomoin.TomographyData.from_dat_file(self._dat_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            written = os.path.join(tmp_dir, 'round_trip.dat')
            original.to_dat_file(written)
            with open(written) as f:
                lines = f.read().splitlines()
            restored = tomoin.TomographyData.from_dat_file(written)
        return original, restored, lines

    def test_round_trip_preserves_parameters(self):
        original, restored, _ = self._round_trip()

        self.assertEqual(restored.parameters, original.parameters,
                         msg='parameters should survive a .dat round trip')

    def test_round_trip_preserves_frames_and_data(self):
        original, restored, _ = self._round_trip()

        self.assertEqual(restored.frames, original.frames,
                         msg='frames should survive a .dat round trip')
        nptest.assert_array_equal(
            restored.frames.raw_data, original.frames.raw_data,
            err_msg='raw data should survive a .dat round trip')

    def test_round_trip_preserves_header(self):
        original, restored, lines = self._round_trip()

        with open(self._dat_path()) as f:
            source = f.read().splitlines()
        self.assertEqual(original.parameters.header,
                         source[:tomoin.DAT_FILE_HEADER_LENGTH],
                         msg='the header lines should be held verbatim')
        self.assertEqual(restored.parameters.header,
                         original.parameters.header,
                         msg='the header lines should be written back')

    def test_comment_lines_are_unchanged(self):
        _, _, lines = self._round_trip()
        with open(self._dat_path()) as f:
            source = f.read().splitlines()

        for i in range(tomoin.PARAMETER_LENGTH):
            if source[i].lstrip().startswith('!'):
                with self.subTest(line=i):
                    self.assertEqual(lines[i], source[i],
                                     msg=f'comment line {i} was altered')

    def test_written_values_hold_the_same_numbers(self):
        _, _, lines = self._round_trip()
        with open(self._dat_path()) as f:
            source = f.read().splitlines()

        for name, i in tomoin.DAT_FILE_LINE_INDEX.items():
            if name in ('raw_data_path', 'output_dir'):
                continue
            with self.subTest(field=name):
                self.assertEqual(float(lines[i]), float(source[i]),
                                 msg=f'{name} changed value when written')

    def test_fortran_indices_are_restored(self):
        original, _, lines = self._round_trip()

        for name in tomoin.FORTRAN_TO_C_FIELDS:
            with self.subTest(field=name):
                written = int(lines[tomoin.DAT_FILE_LINE_INDEX[name]])
                stored = getattr(original.parameters, name)
                self.assertEqual(written, stored + 1,
                                 msg=f'{name} should be written 1-based')

    def test_template_agrees_with_line_index(self):
        lines = tomoin.DAT_FILE_TEMPLATE.splitlines()

        placeholders = {line.strip('{}'): i + tomoin.DAT_FILE_HEADER_LENGTH
                        for i, line in enumerate(lines)
                        if line.startswith('{')}
        self.assertEqual(placeholders, tomoin.DAT_FILE_LINE_INDEX,
                         msg='the write template and the parse indices '
                             'disagree on the file layout')

    def test_mutated_parameters_are_written(self):
        data = tomoin.TomographyData.from_dat_file(self._dat_path())
        data.parameters.niter = 42

        with tempfile.TemporaryDirectory() as tmp_dir:
            written = os.path.join(tmp_dir, 'mutated.dat')
            data.to_dat_file(written)
            restored = tomoin.TomographyData.from_dat_file(written)

        self.assertEqual(restored.parameters.niter, 42,
                         msg='a mutated parameter should be written out')

    def test_absent_value_cannot_be_written(self):
        params = tomoin.Parameters(**PARAMETER_ARGS)

        with self.assertRaises(expt.InputError,
                               msg='a parameter with no value cannot be '
                                   'represented in a .dat file'):
            params.to_dat_lines()

    def test_missing_inline_data_fails(self):
        data = tomoin.TomographyData.from_dat_file(self._dat_path())
        restored = tomoin.TomographyData.model_validate_json(
            data.model_dump_json())

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(expt.RawDataImportError,
                                   msg='writing inline data that is not '
                                       'held should raise an exception'):
                restored.to_dat_file(os.path.join(tmp_dir, 'no_data.dat'))
