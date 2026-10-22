"""Unit-tests for the file_handling module, and the reading and writing of
files through TomographyData.

Run as python test_file_handling.py in console or via coverage
"""

import glob
import io
import json
import os
import tempfile
import unittest
import unittest.mock as mock
import warnings

import numpy.testing as nptest

from .. import commons
from longitudinal_tomography import exceptions as expt
import longitudinal_tomography.utils._input_parsing as parsing
import longitudinal_tomography.utils.file_handling as fhand
import longitudinal_tomography.utils.tomo_data as tdata

base_dir = os.path.split(os.path.realpath(__file__))[0]
resource_dir = os.path.join(base_dir, '..', 'resources')
dat_path = os.path.join(resource_dir, 'INDIVShavingC325.dat')
examples_dir = os.path.join(base_dir, '..', '..', '__EXAMPLES', 'input_files')

MACHINE_ARGS = commons.get_machine_args()

# Machine args that are held by, or derived from, the frames, so have no
# parameter of their own.
DERIVED_MACHINE_ARGS = ('dtbin', 'nprofiles', 'nbins', 'min_dt', 'max_dt')


def _parameter_lines() -> list:
    """The 98 header lines of the reference .dat file, without raw data."""
    infile = os.path.join(resource_dir, 'parameters_INDIVShavingC325.dat')
    with open(infile, 'r') as f:
        return f.readlines()


class TestFileHandling(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _to_json(self, data: tdata.TomographyData) -> tuple[str, dict]:
        json_path = os.path.join(self.tmp_dir, 'data.json')
        data.to_file(json_path)
        with open(json_path) as f:
            return json_path, json.load(f)

    def _from_stdin(self, text: str) -> tdata.TomographyData:
        with mock.patch('sys.stdin', io.StringIO(text)):
            return tdata.TomographyData.from_file('stdin')

    def test_reads_json_from_stdin(self):
        json_path, _ = self._to_json(tdata.TomographyData.from_file(dat_path))
        with open(json_path) as f:
            text = f.read()

        # Blank lines before the document should not matter.
        self.assertEqual(self._from_stdin('\n' + text),
                         tdata.TomographyData.from_file(json_path),
                         msg='JSON should be read from stdin')

    def test_reads_dat_from_stdin(self):
        with open(dat_path) as f:
            text = f.read()

        with mock.patch('sys.stdin', io.StringIO(text + 'after\n')) as stdin:
            data = tdata.TomographyData.from_file('stdin')
            rest = stdin.read()

        self.assertEqual(data, tdata.TomographyData.from_file(dat_path),
                         msg='a .dat file should be read from stdin')
        self.assertEqual(rest, 'after\n',
                         msg='a .dat file should be read from stdin only to '
                             'the end of its raw data')

    def test_reads_dat_with_blank_preamble_from_stdin(self):
        data = tdata.TomographyData.from_file(dat_path)
        data.metadata = tdata.Metadata()
        blank = os.path.join(self.tmp_dir, 'blank.dat')
        data.to_file(blank, fhand.DAT_VERSION)
        with open(blank) as f:
            text = f.read()

        self.assertEqual(self._from_stdin(text),
                         tdata.TomographyData.from_file(blank),
                         msg='a .dat file starting with blank lines should '
                             'be read from stdin')

    def test_dat_file_is_version_2(self):
        self.assertEqual(fhand.detect_version(dat_path), fhand.DAT_VERSION,
                         msg='.dat files should be detected as version 2')

    def test_json_file_is_version_3(self):
        json_path, _ = self._to_json(tdata.TomographyData.from_file(dat_path))
        self.assertEqual(fhand.detect_version(json_path), '3.0.0',
                         msg='JSON files should be written as version 3.0.0')

    def test_read_records_format_version(self):
        data = tdata.TomographyData.from_file(dat_path)
        json_path, _ = self._to_json(data)

        self.assertEqual(data.metadata.format_version, fhand.DAT_VERSION,
                         msg='a .dat file should be recorded as version 2')
        self.assertEqual(
            tdata.TomographyData.from_file(json_path).metadata.format_version,
            '3.0.0', msg='a JSON file should be recorded as version 3.0.0')

    def test_file_without_version_raises(self):
        path = os.path.join(self.tmp_dir, 'empty.json')
        with open(path, 'w') as f:
            f.write('{}')

        with self.assertRaises(expt.InputError,
                               msg='a file with no version should raise'):
            fhand.detect_version(path)

    def test_missing_file_raises(self):
        with self.assertRaises(expt.InputError,
                               msg='a missing input file should raise'):
            tdata.TomographyData.from_file('does/not/exist.dat')

    def test_unsupported_version_raises(self):
        with self.assertRaises(expt.InputError,
                               msg='an unknown version should raise'):
            tdata.TomographyData.from_file(dat_path).to_file(
                os.path.join(self.tmp_dir, 'data.json'), '4.0.0')

    def test_invalid_input_raises(self):
        json_path, document = self._to_json(
            tdata.TomographyData.from_file(dat_path))
        document['frames']['rebin'] = 0
        with open(json_path, 'w') as f:
            json.dump(document, f)

        with self.assertRaises(expt.InputError,
                               msg='a value violating a field constraint '
                                   'should raise an InputError'):
            tdata.TomographyData.from_file(json_path)

    def test_converting_invalid_json_warns(self):
        json_path, document = self._to_json(
            tdata.TomographyData.from_file(dat_path))
        document['frames']['rebin'] = 0
        with open(json_path, 'w') as f:
            json.dump(document, f)
        dat_copy = os.path.join(self.tmp_dir, 'copy.dat')

        with self.assertWarns(UserWarning,
                              msg='converting invalid input should warn'):
            fhand.convert(json_path, dat_copy, fhand.DAT_VERSION)

        with open(dat_copy) as f:
            self.assertEqual(
                f.read().splitlines()[parsing.DAT_FILE_LINE_INDEX['rebin']],
                '0', msg='invalid input should be converted as given')

    def test_converting_invalid_dat_warns(self):
        with open(dat_path) as f:
            lines = f.read().splitlines()
        lines[parsing.DAT_FILE_LINE_INDEX['rebin']] = '0'
        invalid = os.path.join(self.tmp_dir, 'invalid.dat')
        with open(invalid, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        json_path = os.path.join(self.tmp_dir, 'data.json')

        with self.assertWarns(UserWarning,
                              msg='converting invalid input should warn'):
            fhand.convert(invalid, json_path)

        with open(json_path) as f:
            self.assertEqual(json.load(f)['frames']['rebin'], 0,
                             msg='invalid input should be converted as given')

    def test_omitted_defaults_are_written_to_dat(self):
        # An unset output_dir or self-field coefficient cannot be read back
        # from a .dat file.
        kept = ('output_dir', 'g_coupling', 'zwall_over_n')
        omitted = [name for name in fhand._JSON_V3_0_0_DEFAULTS
                   if name not in kept]
        json_path, document = self._to_json(
            tdata.TomographyData.from_file(dat_path))
        for name in omitted:
            del document['parameters'][name]
        with open(json_path, 'w') as f:
            json.dump(document, f)
        dat_copy = os.path.join(self.tmp_dir, 'copy.dat')

        # Omitting a value with a default is valid, so gives no warning.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            fhand.convert(json_path, dat_copy, fhand.DAT_VERSION)

        parameters = tdata.TomographyData.from_file(dat_copy).parameters
        for name in omitted:
            with self.subTest(field=name):
                self.assertEqual(getattr(parameters, name),
                                 fhand._JSON_V3_0_0_DEFAULTS[name],
                                 msg=f'an omitted {name} should be written '
                                     f'as its default')

    def test_whole_floats_are_written_as_integers(self):
        json_path, document = self._to_json(
            tdata.TomographyData.from_file(dat_path))
        document['parameters'].update(niter=20.0, filmstart=0.0)
        document['frames']['rebin'] = 3.0
        document['metadata'].update(
            cycle_stamp_ns=float(document['metadata']['cycle_stamp_ns']),
            c_time=325.0, ring=3.0)
        with open(json_path, 'w') as f:
            json.dump(document, f)
        dat_copy = os.path.join(self.tmp_dir, 'copy.dat')
        fhand.convert(json_path, dat_copy, fhand.DAT_VERSION)

        with open(dat_copy) as f:
            lines = f.read().splitlines()
        for name, expected in (('niter', '20'), ('filmstart', '1'),
                               ('rebin', '3')):
            with self.subTest(field=name):
                self.assertEqual(lines[parsing.DAT_FILE_LINE_INDEX[name]],
                                 expected,
                                 msg=f'a whole float {name} should be '
                                     f'written as an integer')
        # Preamble lines 1, 2 and 4 hold cycle_stamp, c_time and ring.
        self.assertEqual([lines[1], lines[2], lines[4]],
                         ['1520341165', '325', '3'],
                         msg='whole floats in the preamble should be '
                             'written as integers')

    def test_defaults_match_the_models(self):
        # The two match only while 3.0.0 is the latest version.
        defaults = {name: field.default
                    for name, field in tdata.Parameters.model_fields.items()
                    if not field.is_required()}

        self.assertEqual(fhand._JSON_V3_0_0_DEFAULTS, defaults,
                         msg='the version 3.0.0 defaults should be those of '
                             'the models')

    def test_conversion_steps_through_each_version(self):
        steps = {('a', 'b'): lambda text: text + '>b',
                 ('b', 'c'): lambda text: text + '>c',
                 ('c', 'b'): lambda text: text + '<b',
                 ('b', 'a'): lambda text: text + '<a'}
        with mock.patch.object(fhand, 'VERSIONS', ('a', 'b', 'c')), \
                mock.patch.dict(fhand._STEPS, steps, clear=True):
            self.assertEqual(fhand.convert_text('a', 'a', 'c'), 'a>b>c',
                             msg='converting up should step through every '
                                 'version in order')
            self.assertEqual(fhand.convert_text('c', 'c', 'a'), 'c<b<a',
                             msg='converting down should step through '
                                 'every version in order')
            self.assertEqual(fhand.convert_text('b', 'b', 'b'), 'b',
                             msg='converting to the same version should '
                                 'change nothing')

    def test_convert_to_same_version_copies(self):
        dat_copy = os.path.join(self.tmp_dir, 'copy.dat')
        fhand.convert(dat_path, dat_copy, fhand.DAT_VERSION)

        with open(dat_path) as f, open(dat_copy) as g:
            self.assertEqual(g.read(), f.read(),
                             msg='a file converted to its own version '
                                 'should be unchanged')

    def test_dat_and_json_give_the_same_data(self):
        # Equal models give the same Machine and waterfall.
        for path in sorted(glob.glob(os.path.join(examples_dir, '*.dat'))):
            with self.subTest(file=os.path.basename(path)):
                json_path = os.path.join(self.tmp_dir, 'data.json')
                fhand.convert(path, json_path)
                from_dat = tdata.TomographyData.from_file(path)
                from_json = tdata.TomographyData.from_file(json_path)
                # Only the version of the file read should differ.
                from_json.metadata.format_version = fhand.DAT_VERSION
                self.assertEqual(from_json, from_dat,
                                 msg='JSON should hold the same data as the '
                                     '.dat it came from')

    def test_converted_json_matches_written_json(self):
        json_path, _ = self._to_json(tdata.TomographyData.from_file(dat_path))
        converted = os.path.join(self.tmp_dir, 'converted.json')
        fhand.convert(dat_path, converted)

        with open(json_path) as f, open(converted) as g:
            self.assertEqual(g.read(), f.read(),
                             msg='converting a .dat file should give the JSON '
                                 'the models write')

    def test_json_to_dat_to_json_is_unchanged(self):
        json_path, _ = self._to_json(tdata.TomographyData.from_file(dat_path))
        dat_copy = os.path.join(self.tmp_dir, 'copy.dat')
        json_copy = os.path.join(self.tmp_dir, 'copy.json')
        fhand.convert(json_path, dat_copy, fhand.DAT_VERSION)
        fhand.convert(dat_copy, json_copy)

        with open(json_path) as f, open(json_copy) as g:
            self.assertEqual(f.read(), g.read(),
                             msg='JSON should survive a .dat round trip')

    def test_raw_data_is_two_dimensional(self):
        data = tdata.TomographyData.from_file(dat_path)
        _, document = self._to_json(data)
        raw_data = document['frames']['raw_data']

        self.assertEqual(len(raw_data), data.frames.nframes,
                         msg='raw data should hold one row per frame')
        self.assertEqual({len(row) for row in raw_data},
                         {data.frames.nbins_frame},
                         msg='each row should hold one frame')

    def test_raw_data_is_last(self):
        _, document = self._to_json(tdata.TomographyData.from_file(dat_path))

        self.assertEqual(list(document)[-1], 'frames',
                         msg='the frames should be written last')
        self.assertEqual(list(document['frames'])[-1], 'raw_data',
                         msg='the raw data should end the file')

    def test_preamble_is_written(self):
        _, document = self._to_json(tdata.TomographyData.from_file(dat_path))
        metadata = document['metadata']

        expected = {'selector': 'LHCINDIV', 'context_name': 'LHCINDIV',
                    'cycle_stamp_ns': 1520341165 * 10**9, 'c_time': 325}
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(metadata[key], value,
                                 msg='the preamble should be written in the '
                                     'metadata')

    def test_preamble_survives_json(self):
        json_path, _ = self._to_json(tdata.TomographyData.from_file(dat_path))
        dat_copy = os.path.join(self.tmp_dir, 'copy.dat')
        fhand.convert(json_path, dat_copy, fhand.DAT_VERSION)

        with open(dat_path) as f, open(dat_copy) as g:
            self.assertEqual(g.read().splitlines()[:8],
                             f.read().splitlines()[:8],
                             msg='the .dat preamble should survive JSON')

    def test_fitted_synch_part_x_survives(self):
        data = tdata.TomographyData.from_file(dat_path)
        data.parameters.synch_part_x = -1.0
        json_path, _ = self._to_json(data)

        self.assertLess(
            tdata.TomographyData.from_file(json_path).parameters.synch_part_x,
            0, msg='a synch_part_x to fit should stay negative')

    def test_flags_are_written_as_booleans(self):
        _, document = self._to_json(tdata.TomographyData.from_file(dat_path))
        parameters = document['parameters']

        for flag in ('full_pp_flag', 'self_field_flag'):
            with self.subTest(flag=flag):
                self.assertIs(type(parameters[flag]), bool,
                              msg=f'{flag} should be written as a boolean')

    def test_metadata_survives_json(self):
        data = tdata.TomographyData.from_file(dat_path)
        data.metadata = tdata.Metadata(
            format_version='3.0.0', selector='CPS.USER.MD3',
            cycle_stamp_ns=1, context_name='MD', acquisition_id=4,
            bunch_id=2)
        json_path, _ = self._to_json(data)

        self.assertEqual(tdata.TomographyData.from_file(json_path).metadata,
                         data.metadata,
                         msg='metadata should survive a JSON round trip')

    def test_parameter_descriptions_survive_json(self):
        data = tdata.TomographyData.from_file(dat_path)
        data.parameter_descriptions = {'vrf1': 'RF'}
        json_path, _ = self._to_json(data)

        self.assertEqual(
            tdata.TomographyData.from_file(json_path).parameter_descriptions,
            {'vrf1': 'RF'},
            msg='descriptions should survive a JSON round trip')

    def test_wrong_raw_data_shape_raises(self):
        json_path, document = self._to_json(
            tdata.TomographyData.from_file(dat_path))
        frames = document['frames']
        frames['raw_data'] = [sum(frames['raw_data'], [])]
        with open(json_path, 'w') as f:
            json.dump(document, f)

        with self.assertRaises(expt.RawDataImportError,
                               msg='raw data not shaped by frame should '
                                   'raise'):
            tdata.TomographyData.from_file(json_path)


class TestDatParsing(unittest.TestCase):
    """Conversion of the lines of a version 2 `.dat` file to version 3.0.0.
    """

    def _document(self, lines: list) -> dict:
        return json.loads(fhand.dat_to_json(lines))

    def _parameters(self, lines: list) -> tdata.Parameters:
        return tdata.Parameters(**self._document(lines)['parameters'])

    def test_gives_latest_version(self):
        metadata = self._document(_parameter_lines())['metadata']

        self.assertEqual(metadata['format_version'], fhand.JSON_VERSION,
                         msg='.dat lines should be converted to the latest '
                             'version')

    def test_advances_through_later_versions(self):
        with mock.patch.object(fhand, 'VERSIONS', ('2', '3.0.0', '4')), \
                mock.patch.object(fhand, 'JSON_VERSION', '4'), \
                mock.patch.dict(fhand._STEPS,
                                {('3.0.0', '4'): lambda text: 'version 4'}):
            self.assertEqual(fhand.dat_to_json(_parameter_lines()),
                             'version 4',
                             msg='.dat lines should be converted through '
                                 'every later version')

    def test_matches_machine_args(self):
        params = self._parameters(_parameter_lines())

        for name, expected in MACHINE_ARGS.items():
            if name in DERIVED_MACHINE_ARGS:
                continue
            self.assertEqual(
                getattr(params, name), expected,
                msg=f'{name} parsed incorrectly from the .dat file')

    def test_frame_values_are_held_by_frames(self):
        document = self._document(_parameter_lines())
        parameters, frames = document['parameters'], document['frames']

        for frame_name, name, value in (
                ('nframes', 'framecount', 150),
                ('nbins_frame', 'framelength', 1000),
                ('skip_frames', 'skip_frames', 0),
                ('skip_bins_start', 'skip_bins_start', 170),
                ('skip_bins_end', 'skip_bins_end', 70),
                ('rebin', 'rebin', 3),
                ('sampling_time', 'dtbin', 9.999999999999999E-10)):
            with self.subTest(field=frame_name):
                self.assertEqual(frames[frame_name], value,
                                 msg=f'{frame_name} not taken from {name}')
                self.assertNotIn(name, parameters,
                                 msg=f'{name} should be held only by the '
                                     f'frames')
        self.assertEqual(frames['raw_data_path'], 'pipe',
                         msg='raw_data_path not taken from the .dat file')
        self.assertIsNone(frames['raw_data'],
                          msg='no raw data should be held unless given')

    def test_converts_fortran_indices(self):
        lines = _parameter_lines()
        for name, value in (('filmstart', 3), ('filmstop', 5),
                            ('beam_ref_frame', 7), ('machine_ref_frame', 9)):
            lines[parsing.DAT_FILE_LINE_INDEX[name]] = f'{value}\n'
        # dturns is an ordinary int field, it must survive untouched.
        lines[parsing.DAT_FILE_LINE_INDEX['dturns']] = '9\n'

        params = self._parameters(lines)

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

    def test_leaves_input_unchanged(self):
        lines = _parameter_lines()
        original = list(lines)

        fhand.dat_to_json(lines)

        self.assertEqual(lines, original,
                         msg='the conversion must not mutate its input')

    def test_strips_line_endings(self):
        lines = [line.rstrip('\n') + '\r\n' for line in _parameter_lines()]

        params = self._parameters(lines)

        self.assertEqual(params.output_dir, MACHINE_ARGS['output_dir'],
                         msg='carriage returns not stripped from .dat lines')

    def test_strips_padding(self):
        # Fortran writers pad numeric fields.
        lines = [f'   {line.strip()}  \n' for line in _parameter_lines()]

        self.assertEqual(fhand.dat_to_json(lines),
                         fhand.dat_to_json(_parameter_lines()),
                         msg='padded .dat lines should parse identically to '
                             'unpadded ones')

    def test_flags_parsed_from_dat_digits(self):
        for value, expected in (('0', False), ('1', True)):
            with self.subTest(value=value):
                lines = _parameter_lines()
                lines[parsing.DAT_FILE_LINE_INDEX['full_pp_flag']] = \
                    f'{value}\n'
                lines[parsing.DAT_FILE_LINE_INDEX['self_field_flag']] = \
                    f'{value}\n'

                params = self._parameters(lines)

                self.assertIs(params.full_pp_flag, expected,
                              msg='full_pp_flag parsed incorrectly')
                self.assertIs(params.self_field_flag, expected,
                              msg='self_field_flag parsed incorrectly')

    def test_preamble_is_parsed(self):
        metadata = self._document(_parameter_lines())['metadata']

        expected = {'selector': 'LHCINDIV', 'context_name': 'LHCINDIV',
                    'cycle_stamp_ns': 1520341165 * 10**9, 'c_time': 325,
                    'timespan': 1015.943, 'ring': 3,
                    'delay': 1259.0000000000002, 'v_scale': 0.01,
                    'v_position': -1.8679245283018875}
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertEqual(metadata[name], value,
                                 msg=f'{name} should be read from the '
                                     f'preamble')

    def test_unparsable_index_fails(self):
        for name in parsing.FORTRAN_TO_C_FIELDS:
            with self.subTest(field=name):
                lines = _parameter_lines()
                lines[parsing.DAT_FILE_LINE_INDEX[name]] = 'not a number\n'

                with self.assertRaises(
                        expt.InputError,
                        msg=f'an uncoercible {name} should raise '
                            f'an InputError'):
                    fhand.dat_to_json(lines)

    def test_unparsable_value_fails(self):
        lines = _parameter_lines()
        lines[parsing.DAT_FILE_LINE_INDEX['dturns']] = 'not a number\n'

        with self.assertRaises(expt.InputError,
                               msg='a value that cannot be coerced should '
                                   'raise an InputError'):
            fhand.dat_to_json(lines)

    def test_no_iter_fails(self):
        with self.assertRaises(expt.InputError,
                               msg='non iterable input should '
                                   'raise an exception'):
            fhand.dat_to_json(1)

    def test_wrong_len_fails(self):
        with self.assertRaises(expt.InputError,
                               msg='input of the wrong length should '
                                   'raise an exception'):
            fhand.dat_to_json(['1', '2', '3'])


class TestDatWriting(unittest.TestCase):
    """Writing input data as a version 2 `.dat` file."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _write_dat(self, data: tdata.TomographyData) -> tuple[str, list]:
        written = os.path.join(self.tmp_dir, 'written.dat')
        data.to_file(written, fhand.DAT_VERSION)
        with open(written) as f:
            return written, f.read().splitlines()

    def _round_trip(self) -> tuple:
        original = tdata.TomographyData.from_file(dat_path)
        written, lines = self._write_dat(original)
        restored = tdata.TomographyData.from_file(written)
        return original, restored, lines

    def _source_lines(self) -> list:
        with open(dat_path) as f:
            return f.read().splitlines()

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

    def test_round_trip_preserves_preamble(self):
        original, restored, lines = self._round_trip()

        preamble = self._source_lines()[:parsing.DAT_FILE_HEADER_LENGTH]
        self.assertEqual(lines[:parsing.DAT_FILE_HEADER_LENGTH], preamble,
                         msg='the preamble should be written back')
        self.assertEqual(restored.metadata, original.metadata,
                         msg='the metadata should survive a .dat round trip')

    def test_empty_metadata_gives_blank_preamble(self):
        data = tdata.TomographyData.from_file(dat_path)
        data.metadata = tdata.Metadata()
        _, lines = self._write_dat(data)

        self.assertEqual(lines[:parsing.DAT_FILE_HEADER_LENGTH],
                         [''] * parsing.DAT_FILE_HEADER_LENGTH,
                         msg='metadata without values should give blank '
                             'lines')

    def test_comment_lines_are_unchanged(self):
        _, _, lines = self._round_trip()
        source = self._source_lines()

        for i in range(parsing.PARAMETER_LENGTH):
            if source[i].lstrip().startswith('!'):
                with self.subTest(line=i):
                    self.assertEqual(lines[i], source[i],
                                     msg=f'comment line {i} was altered')

    def test_written_values_hold_the_same_numbers(self):
        _, _, lines = self._round_trip()
        source = self._source_lines()

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
        data = tdata.TomographyData.from_file(dat_path)
        data.parameters.niter = 42
        written, _ = self._write_dat(data)

        self.assertEqual(
            tdata.TomographyData.from_file(written).parameters.niter, 42,
            msg='a mutated parameter should be written out')

    def test_absent_output_dir_cannot_be_written(self):
        data = tdata.TomographyData.from_file(dat_path)
        data.parameters.output_dir = None

        with self.assertRaises(expt.InputError,
                               msg='an output directory with no value '
                                   'cannot be represented in a .dat file'):
            self._write_dat(data)

    def test_absent_optional_is_written_as_zero(self):
        optional = ('g_coupling', 'zwall_over_n')
        data = tdata.TomographyData.from_file(dat_path)
        for name in optional:
            setattr(data.parameters, name, None)
        _, lines = self._write_dat(data)

        for name in optional:
            with self.subTest(field=name):
                self.assertEqual(lines[parsing.DAT_FILE_LINE_INDEX[name]], '0',
                                 msg=f'{name} should be written as zero')

    def test_missing_inline_data_fails(self):
        data = tdata.TomographyData.from_file(dat_path)
        data.frames.raw_data = None

        with self.assertRaises(expt.RawDataImportError,
                               msg='writing inline data that is not '
                                   'held should raise an exception'):
            self._write_dat(data)

    def test_data_is_written_inline_without_a_path(self):
        data = tdata.TomographyData.from_file(dat_path)
        data.frames.raw_data_path = ''
        written, lines = self._write_dat(data)

        self.assertEqual(lines[parsing.RAW_DATA_FILE_IDX], 'pipe',
                         msg='frames naming no raw-data file should be '
                             'written inline')
        nptest.assert_array_equal(
            tdata.TomographyData.from_file(written).frames.raw_data,
            data.frames.raw_data,
            err_msg='the raw data written inline should be read back')

    def test_missing_inline_data_without_a_path_fails(self):
        data = tdata.TomographyData.from_file(dat_path)
        data.frames.raw_data_path = ''
        data.frames.raw_data = None

        with self.assertRaises(expt.RawDataImportError,
                               msg='writing frames that hold neither '
                                   'data nor a path should raise'):
            self._write_dat(data)
