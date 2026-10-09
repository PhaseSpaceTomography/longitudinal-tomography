"""Unit-tests for the tomo_data module.

Run as python test_tomo_data.py in console or via coverage
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
import longitudinal_tomography.utils._input_parsing as parsing
import longitudinal_tomography.utils.tomo_data as tdata
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
        params = tdata.Parameters.from_dat_lines(_parameter_lines())

        for name, expected in MACHINE_ARGS.items():
            if name in DERIVED_MACHINE_ARGS:
                continue
            self.assertEqual(
                getattr(params, name), expected,
                msg=f'{name} parsed incorrectly from the .dat file')

    def test_from_dat_lines_frame_fields(self):
        params = tdata.Parameters.from_dat_lines(_parameter_lines())

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
            lines[parsing.DAT_FILE_LINE_INDEX[name]] = f'{value}\n'
        # dturns is an ordinary int field, it must survive untouched.
        lines[parsing.DAT_FILE_LINE_INDEX['dturns']] = '9\n'

        params = tdata.Parameters.from_dat_lines(lines)

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

        tdata.Parameters.from_dat_lines(lines)

        self.assertEqual(lines, original,
                         msg='from_dat_lines must not mutate its input')

    def test_from_dat_lines_strips_line_endings(self):
        lines = [line.rstrip('\n') + '\r\n' for line in _parameter_lines()]

        params = tdata.Parameters.from_dat_lines(lines)

        self.assertEqual(params.output_dir, MACHINE_ARGS['output_dir'],
                         msg='carriage returns not stripped from .dat lines')

    def test_from_dat_lines_strips_padding(self):
        # Fortran writers pad numeric fields; bool coercion of a padded
        # value fails unless the whole line is stripped.
        lines = [f'   {line.strip()}  \n' for line in _parameter_lines()]

        params = tdata.Parameters.from_dat_lines(lines)
        expected = tdata.Parameters.from_dat_lines(_parameter_lines())

        self.assertEqual(params.model_dump(), expected.model_dump(),
                         msg='padded .dat lines should parse identically to '
                             'unpadded ones')

    def test_from_dat_lines_unparsable_index_fails(self):
        # The index fields are converted before pydantic sees them, so they
        # need their own error handling.
        for name in parsing.FORTRAN_TO_C_FIELDS:
            with self.subTest(field=name):
                lines = _parameter_lines()
                lines[parsing.DAT_FILE_LINE_INDEX[name]] = 'not a number\n'

                with self.assertRaises(
                        expt.InputError,
                        msg=f'an uncoercible {name} should raise '
                            f'an InputError'):
                    tdata.Parameters.from_dat_lines(lines)

    def test_defaults_match_machine_defaults(self):
        defaults = {**machine_base._machine_base_defaults,
                    **mch._machine_opts_def}

        for name, expected in defaults.items():
            if name not in tdata.Parameters.model_fields:
                continue
            with self.subTest(field=name):
                self.assertEqual(
                    tdata.Parameters.model_fields[name].default, expected,
                    msg=f'the default for {name} differs from the one used '
                        f'by Machine')

    def test_validate_rejects_an_invalid_change(self):
        params = tdata.Parameters.from_dat_lines(_parameter_lines())
        params.niter = -1

        with self.assertRaises(pydantic.ValidationError,
                               msg='an invalid value should be rejected by '
                                   'validate'):
            params.validate()

    def test_validate_rejects_a_broken_cross_field_bound(self):
        params = tdata.Parameters.from_dat_lines(_parameter_lines())
        params.filmstop = params.framecount

        with self.assertRaises(pydantic.ValidationError,
                               msg='filmstop beyond framecount should be '
                                   'rejected by validate'):
            params.validate()

    def test_from_dat_lines_no_iter_fails(self):
        with self.assertRaises(expt.InputError,
                               msg='non iterable input should '
                                   'raise an exception'):
            tdata.Parameters.from_dat_lines(1)

    def test_from_dat_lines_wrong_len_fails(self):
        with self.assertRaises(expt.InputError,
                               msg='input of the wrong length should '
                                   'raise an exception'):
            tdata.Parameters.from_dat_lines(['1', '2', '3'])

    def test_from_dat_lines_invalid_value_fails(self):
        lines = _parameter_lines()
        lines[parsing.DAT_FILE_LINE_INDEX['rebin']] = '0\n'

        with self.assertRaises(expt.InputError,
                               msg='a value violating a field constraint '
                                   'should raise an InputError'):
            tdata.Parameters.from_dat_lines(lines)

    def test_from_dat_lines_unparsable_value_fails(self):
        lines = _parameter_lines()
        lines[parsing.DAT_FILE_LINE_INDEX['dturns']] = 'not a number\n'

        with self.assertRaises(expt.InputError,
                               msg='a value that cannot be coerced should '
                                   'raise an InputError'):
            tdata.Parameters.from_dat_lines(lines)

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
                    tdata.Parameters(**{**PARAMETER_ARGS, name: value})

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
                    tdata.Parameters(**{**PARAMETER_ARGS, name: value})

    def test_cross_field_violations_all_raised(self):
        bad_input = {'skip_frames':150, 'skip_bins_start': 1000,
                     'skip_bins_end': 1000, 'synch_part_x': 1000.0,
                     'filmstart': 150, 'filmstop': 150,
                     'beam_ref_frame': 150, 'machine_ref_frame': 150}
        with (self.assertRaises(pydantic.ValidationError,
                               msg="Ill formed input should be rejected")
                               as exception):
            tdata.Parameters(**{**PARAMETER_ARGS, **bad_input})

        for key in bad_input.keys():
            self.assertIn(key, str(exception.exception))

    def test_accepts_numpy_scalars(self):
        params = tdata.Parameters(
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
            tdata.Parameters(**{**PARAMETER_ARGS, 'dturns': 5.5})

    def test_flags_parsed_from_dat_digits(self):
        for value, expected in (('0', False), ('1', True)):
            with self.subTest(value=value):
                lines = _parameter_lines()
                lines[parsing.DAT_FILE_LINE_INDEX['full_pp_flag']] = \
                    f'{value}\n'
                lines[parsing.DAT_FILE_LINE_INDEX['self_field_flag']] = \
                    f'{value}\n'

                params = tdata.Parameters.from_dat_lines(lines)

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
                    tdata.Frames(**{**frame_input_args, name: value})

    def test_rejects_cross_field_violations(self):
        for name, value in (('skip_frames', 150), ('skip_bins_start', 1000),
                            ('skip_bins_end', 1000)):
            with self.subTest(field=name):
                with self.assertRaises(
                        pydantic.ValidationError,
                        msg=f'{name}={value} exceeds its frame bound and '
                            f'should be rejected'):
                    tdata.Frames(**{**frame_input_args, name: value})

    def test_validate_accepts_a_valid_change(self):
        frames = tdata.Frames(**frame_input_args)
        frames.skip_frames = 5

        frames.validate()

        self.assertEqual(frames.skip_frames, 5,
                         msg='a valid value should be assignable')

    def test_validate_rejects_an_invalid_change(self):
        for name, value in (('skip_frames', -1), ('nframes', 0),
                            ('rebin', 0), ('skip_bins_start', 1000)):
            with self.subTest(field=name):
                frames = tdata.Frames(**frame_input_args)
                setattr(frames, name, value)
                with self.assertRaises(
                        pydantic.ValidationError,
                        msg=f'{name}={value} should be rejected by '
                            f'validate'):
                    frames.validate()

    def test_validate_rejects_bad_raw_data(self):
        frames = tdata.Frames(**frame_input_args)
        frames.raw_data = [1.0]

        with self.assertRaises(expt.RawDataImportError,
                               msg='raw data of the wrong length should be '
                                   'rejected by validate'):
            frames.validate()

    def test_from_parameters_maps_fields(self):
        params = tdata.Parameters.from_dat_lines(_parameter_lines())
        frames = tdata.Frames.from_parameters(params, raw_data_path='a/path')

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
        params = tdata.Parameters.from_dat_lines(_parameter_lines())
        frames = tdata.Frames.from_parameters(params)

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
        positional = tdata.Frames(
            frame_input_args['framecount'], frame_input_args['framelength'],
            frame_input_args['skip_frames'],
            frame_input_args['skip_bins_start'],
            frame_input_args['skip_bins_end'], frame_input_args['rebin'],
            frame_input_args['dtbin'])

        self.assertEqual(positional, tdata.Frames(**frame_input_args),
                         msg='positional and legacy keyword construction '
                             'should agree')

    def test_field_names_accepted(self):
        self.assertEqual(tdata.Frames(**self._field_args()),
                         tdata.Frames(**frame_input_args),
                         msg='construction by field name and by legacy name '
                             'should agree')

    def test_too_many_positional_args_fails(self):
        with self.assertRaises(TypeError,
                               msg='more positional arguments than the '
                                   'legacy signature takes should fail'):
            tdata.Frames(150, 1000, 0, 170, 70, 3, 1e-9, '', 'extra')

    def test_duplicate_argument_fails(self):
        with self.assertRaises(TypeError,
                               msg='the same argument given positionally and '
                                   'by keyword should fail'):
            tdata.Frames(150, 1000, 0, 170, 70, 3, 1e-9, framecount=150)

    def test_model_validate_round_trip(self):
        frames = tdata.Frames(**frame_input_args)

        self.assertEqual(tdata.Frames.model_validate(frames.model_dump()),
                         frames,
                         msg='a Frames should survive a dump/validate cycle')

    def test_equality_compares_raw_data(self):
        a = tdata.Frames(**frame_input_args)
        b = tdata.Frames(**frame_input_args)
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

    def test_raw_data_accepted_at_construction(self):
        raw_data = np.arange(150 * 1000)

        frames = tdata.Frames(**frame_input_args, raw_data=raw_data)

        nptest.assert_array_equal(
            frames.raw_data, raw_data,
            err_msg='raw data given at construction should be held')

    def test_raw_data_coerced_to_array(self):
        values = [float(i) for i in range(150 * 1000)]

        frames = tdata.Frames(**frame_input_args, raw_data=values)

        self.assertIsInstance(frames.raw_data, np.ndarray,
                              msg='a list of raw data should be coerced to '
                                  'an array')

    def test_raw_data_wrong_length_at_construction_fails(self):
        with self.assertRaises(expt.RawDataImportError,
                               msg='raw data of the wrong length should '
                                   'raise an exception'):
            tdata.Frames(**frame_input_args, raw_data=[1.0])

    def test_raw_data_dumped_as_list(self):
        frames = tdata.Frames(**frame_input_args, raw_data=np.zeros(150000))

        self.assertIsInstance(frames.model_dump()['raw_data'], list,
                              msg='raw data should dump as a list')

    def test_equality_with_other_type(self):
        self.assertNotEqual(tdata.Frames(**frame_input_args), 1,
                            msg='a Frames should not equal a non-Frames')


class TestTomographyData(unittest.TestCase):

    def _dat_path(self) -> str:
        return os.path.join(_resource_dir(), 'INDIVShavingC325.dat')

    def test_from_dat_file_parameters(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())

        self.assertIsInstance(data.parameters, tdata.Parameters,
                              msg='parameters should be a Parameters')
        self.assertIsInstance(data.frames, tdata.Frames,
                              msg='frames should be a Frames')

        expected = tdata.Parameters.from_dat_lines(_parameter_lines())
        self.assertEqual(
            data.parameters.model_dump(), expected.model_dump(),
            msg='parameters read from the file differ from the same file '
                'parsed as lines')

    def test_from_dat_file_frames(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())

        self.assertEqual(data.frames.nprofs, 150,
                         msg='nprofs not derived from the parsed parameters')
        self.assertEqual(data.frames.nbins, 760,
                         msg='nbins not derived from the parsed parameters')
        self.assertEqual(data.frames.raw_data_path, 'pipe',
                         msg='raw_data_path not taken from the .dat file')

    def test_from_dat_file_holds_raw_data(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())

        self.assertIsNotNone(data.frames.raw_data,
                             msg='raw data should be attached to the frames')
        self.assertEqual(len(data.frames.raw_data), 150 * 1000,
                         msg='wrong amount of raw data attached')

    def test_from_dat_file_waterfall_correct(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())

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

        data = tdata.TomographyData.from_dat_file(self._dat_path())

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
        self.assertEqual(data.frames.model_dump(exclude={'raw_data'}),
                         frames.model_dump(exclude={'raw_data'}),
                         msg='frames differ between TomographyData and '
                             'txt_input_to_machine')
        self.assertIsNone(frames.raw_data,
                          msg='txt_input_to_machine should not attach '
                              'raw data')

    def test_json_round_trip(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())

        restored = tdata.TomographyData.model_validate_json(
            data.model_dump_json())

        self.assertEqual(restored.parameters, data.parameters,
                         msg='parameters should survive a JSON round trip')
        self.assertEqual(restored.frames, data.frames,
                         msg='frames should survive a JSON round trip')
        nptest.assert_array_equal(
            restored.frames.raw_data, data.frames.raw_data,
            err_msg='raw data should survive a JSON round trip')

    def test_validate_checks_nested_models(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())
        data.validate()

        data.parameters.niter = -1

        with self.assertRaises(pydantic.ValidationError,
                               msg='validate should reach the parameters it '
                                   'holds'):
            data.validate()

    def test_validate_checks_nested_frames(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())
        data.frames.rebin = 0

        with self.assertRaises(pydantic.ValidationError,
                               msg='validate should reach the frames it '
                                   'holds'):
            data.validate()

    def test_from_dat_file_missing_file_fails(self):
        with self.assertRaises(expt.InputError,
                               msg='a missing input file should raise '
                                   'an exception'):
            tdata.TomographyData.from_dat_file('does/not/exist.dat')


class TestDatFileWriting(unittest.TestCase):

    def _dat_path(self) -> str:
        return os.path.join(_resource_dir(), 'INDIVShavingC325.dat')

    def _round_trip(self) -> tuple:
        original = tdata.TomographyData.from_dat_file(self._dat_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            written = os.path.join(tmp_dir, 'round_trip.dat')
            original.to_dat_file(written)
            with open(written) as f:
                lines = f.read().splitlines()
            restored = tdata.TomographyData.from_dat_file(written)
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
                         source[:parsing.DAT_FILE_HEADER_LENGTH],
                         msg='the header lines should be held verbatim')
        self.assertEqual(restored.parameters.header,
                         original.parameters.header,
                         msg='the header lines should be written back')

    def test_comment_lines_are_unchanged(self):
        _, _, lines = self._round_trip()
        with open(self._dat_path()) as f:
            source = f.read().splitlines()

        for i in range(parsing.PARAMETER_LENGTH):
            if source[i].lstrip().startswith('!'):
                with self.subTest(line=i):
                    self.assertEqual(lines[i], source[i],
                                     msg=f'comment line {i} was altered')

    def test_written_values_hold_the_same_numbers(self):
        _, _, lines = self._round_trip()
        with open(self._dat_path()) as f:
            source = f.read().splitlines()

        for name, i in parsing.DAT_FILE_LINE_INDEX.items():
            if name in ('raw_data_path', 'output_dir'):
                continue
            with self.subTest(field=name):
                self.assertEqual(float(lines[i]), float(source[i]),
                                 msg=f'{name} changed value when written')

    def test_fortran_indices_are_restored(self):
        original, _, lines = self._round_trip()

        for name in parsing.FORTRAN_TO_C_FIELDS:
            with self.subTest(field=name):
                written = int(lines[parsing.DAT_FILE_LINE_INDEX[name]])
                stored = getattr(original.parameters, name)
                self.assertEqual(written, stored + 1,
                                 msg=f'{name} should be written 1-based')

    def test_template_agrees_with_line_index(self):
        lines = parsing.DAT_FILE_TEMPLATE.splitlines()

        placeholders = {line.strip('{}'): i + parsing.DAT_FILE_HEADER_LENGTH
                        for i, line in enumerate(lines)
                        if line.startswith('{')}
        self.assertEqual(placeholders, parsing.DAT_FILE_LINE_INDEX,
                         msg='the write template and the parse indices '
                             'disagree on the file layout')

    def test_mutated_parameters_are_written(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())
        data.parameters.niter = 42

        with tempfile.TemporaryDirectory() as tmp_dir:
            written = os.path.join(tmp_dir, 'mutated.dat')
            data.to_dat_file(written)
            restored = tdata.TomographyData.from_dat_file(written)

        self.assertEqual(restored.parameters.niter, 42,
                         msg='a mutated parameter should be written out')

    def test_absent_output_dir_cannot_be_written(self):
        params = tdata.Parameters(**PARAMETER_ARGS)

        with self.assertRaises(expt.InputError,
                               msg='an output directory with no value '
                                   'cannot be represented in a .dat file'):
            params.to_dat_lines()

    def test_absent_optional_is_written_as_zero(self):
        optional = ('g_coupling', 'zwall_over_n', 'pickup_sensitivity')
        params = tdata.Parameters(**PARAMETER_ARGS, output_dir='/tmp/')
        lines = params.to_dat_lines()

        for name in optional:
            with self.subTest(field=name):
                self.assertIsNone(getattr(params, name),
                                  msg=f'{name} should have no value')
                self.assertEqual(lines[parsing.DAT_FILE_LINE_INDEX[name]], '0',
                                 msg=f'{name} should be written as zero')

    def test_missing_inline_data_fails(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())
        without_data = tdata.TomographyData(
            parameters=data.parameters,
            frames=tdata.Frames.from_parameters(data.parameters,
                                                 raw_data_path='pipe'))

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(expt.RawDataImportError,
                                   msg='writing inline data that is not '
                                       'held should raise an exception'):
                without_data.to_dat_file(
                    os.path.join(tmp_dir, 'no_data.dat'))

    def test_data_is_written_inline_without_a_path(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())
        no_path = tdata.TomographyData(
            parameters=data.parameters,
            frames=tdata.Frames.from_parameters(
                data.parameters, raw_data=data.frames.raw_data))

        with tempfile.TemporaryDirectory() as tmp_dir:
            written = os.path.join(tmp_dir, 'no_path.dat')
            no_path.to_dat_file(written)
            restored = tdata.TomographyData.from_dat_file(written)

        nptest.assert_array_equal(
            restored.frames.raw_data, data.frames.raw_data,
            err_msg='frames naming no raw-data file should be written inline')

    def test_missing_inline_data_without_a_path_fails(self):
        data = tdata.TomographyData.from_dat_file(self._dat_path())
        without_data = tdata.TomographyData(
            parameters=data.parameters,
            frames=tdata.Frames.from_parameters(data.parameters))

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(expt.RawDataImportError,
                                   msg='writing frames that hold neither '
                                       'data nor a path should raise'):
                without_data.to_dat_file(
                    os.path.join(tmp_dir, 'no_data.dat'))
