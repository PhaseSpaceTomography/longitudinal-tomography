"""Unit-tests for the tomo_data module.

Run as python test_tomo_data.py in console or via coverage
"""

import os
import unittest

import numpy as np
import numpy.testing as nptest
import pydantic

from .. import commons
import longitudinal_tomography.tracking.machine as mch
import longitudinal_tomography.tracking.machine_base as machine_base
from longitudinal_tomography import exceptions as expt
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


def _reference_data() -> tdata.TomographyData:
    """The input data of the reference .dat file."""
    return tdata.TomographyData.from_file(
        os.path.join(_resource_dir(), 'INDIVShavingC325.dat'))


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
        params = _reference_data().parameters
        params.niter = -1

        with self.assertRaises(pydantic.ValidationError,
                               msg='an invalid value should be rejected by '
                                   'validate'):
            params.validate()

    def test_validate_rejects_a_broken_cross_field_bound(self):
        params = _reference_data().parameters
        params.filmstop = params.framecount

        with self.assertRaises(pydantic.ValidationError,
                               msg='filmstop beyond framecount should be '
                                   'rejected by validate'):
            params.validate()

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

    def test_rejects_cutting_every_bin(self):
        with self.assertRaises(pydantic.ValidationError,
                               msg='cutting every bin of a frame should be '
                                   'rejected'):
            tdata.Parameters(**{**PARAMETER_ARGS, 'skip_bins_start': 500,
                                'skip_bins_end': 500})

    def test_indices_bounded_by_profiles(self):
        # With 10 of 150 frames skipped, there are 140 profiles.
        for name in ('filmstart', 'filmstop', 'beam_ref_frame',
                     'machine_ref_frame'):
            with self.subTest(field=name):
                args = {**PARAMETER_ARGS, 'skip_frames': 10, 'filmstop': 139}
                tdata.Parameters(**{**args, name: 139})
                with self.assertRaises(
                        pydantic.ValidationError,
                        msg=f'{name} beyond the last profile should be '
                            f'rejected'):
                    tdata.Parameters(**{**args, name: 140})

    def test_self_fields_need_coefficients(self):
        coefficients = {'g_coupling': 0.0, 'zwall_over_n': 0.0}
        tdata.Parameters(**PARAMETER_ARGS, **coefficients,
                         self_field_flag=True)

        for name in coefficients:
            with self.subTest(field=name):
                with self.assertRaises(
                        pydantic.ValidationError,
                        msg=f'self-fields without {name} should be '
                            f'rejected'):
                    tdata.Parameters(**PARAMETER_ARGS,
                                     **{**coefficients, name: None},
                                     self_field_flag=True)

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

    def test_rejects_cutting_every_bin(self):
        with self.assertRaises(pydantic.ValidationError,
                               msg='cutting every bin of a frame should be '
                                   'rejected'):
            tdata.Frames(**{**frame_input_args, 'skip_bins_start': 500,
                            'skip_bins_end': 500})

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
        params = _reference_data().parameters
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
        params = _reference_data().parameters
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

    def test_from_file_parameters(self):
        data = _reference_data()

        self.assertIsInstance(data.parameters, tdata.Parameters,
                              msg='parameters should be a Parameters')
        self.assertIsInstance(data.frames, tdata.Frames,
                              msg='frames should be a Frames')

        for name, expected in MACHINE_ARGS.items():
            if name in DERIVED_MACHINE_ARGS:
                continue
            with self.subTest(field=name):
                self.assertEqual(getattr(data.parameters, name), expected,
                                 msg=f'{name} read incorrectly from the file')

    def test_from_file_frames(self):
        data = _reference_data()

        self.assertEqual(data.frames.nprofs, 150,
                         msg='nprofs not derived from the parsed parameters')
        self.assertEqual(data.frames.nbins, 760,
                         msg='nbins not derived from the parsed parameters')
        self.assertEqual(data.frames.raw_data_path, 'pipe',
                         msg='raw_data_path not taken from the .dat file')

    def test_from_file_holds_raw_data(self):
        data = _reference_data()

        self.assertIsNotNone(data.frames.raw_data,
                             msg='raw data should be attached to the frames')
        self.assertEqual(len(data.frames.raw_data), 150 * 1000,
                         msg='wrong amount of raw data attached')

    def test_from_file_waterfall_correct(self):
        data = _reference_data()

        waterfall = data.frames.to_waterfall(data.frames.raw_data)

        machine = mch.Machine(**MACHINE_ARGS)
        profiles = tomoin.raw_data_to_profiles(
            waterfall, machine, data.frames.rebin, data.frames.sampling_time)

        nptest.assert_almost_equal(
            profiles.waterfall, commons.load_waterfall(),
            err_msg='waterfall built from TomographyData does not match the '
                    'reference waterfall')

    def test_from_file_agrees_with_txt_input_to_machine(self):
        raw_params, _ = tomoin.get_user_input(self._dat_path())
        machine, frames = tomoin.txt_input_to_machine(raw_params)

        data = _reference_data()

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
        data = _reference_data()

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
        data = _reference_data()
        data.validate()

        data.parameters.niter = -1

        with self.assertRaises(pydantic.ValidationError,
                               msg='validate should reach the parameters it '
                                   'holds'):
            data.validate()

    def test_validate_checks_nested_frames(self):
        data = _reference_data()
        data.frames.rebin = 0

        with self.assertRaises(pydantic.ValidationError,
                               msg='validate should reach the frames it '
                                   'holds'):
            data.validate()
