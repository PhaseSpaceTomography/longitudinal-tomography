"""Reading tomography input files and converting them between format
versions.

Each format version is converted to the next and to the previous by its own
function, and converting between two versions chains those functions through
every version in between. The functions work on the text of a file, without
the pydantic models, which follow only the latest version. Legacy `.dat`
files are format version 2, JSON files version 3.0.0.

:Author(s): **Simon Albright**
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING

import numpy as np

from .. import exceptions as expt
from ..utils import _input_parsing as parsing, parameter_descriptions as pdesc

if TYPE_CHECKING:
    from typing import Sequence
    from numpy.typing import NDArray as NPArray

#: Every format version, oldest first.
VERSIONS = ('2', '3.0.0')
#: Format version of legacy `.dat` files.
DAT_VERSION = '2'
#: Latest format version, the one the pydantic models read and write, and the
#: default for converting.
JSON_VERSION = VERSIONS[-1]

# Version 2 values held as integers or flags in version 3.0.0; all others but
# output_dir are floats. cycle_stamp is preamble line 1, in seconds.
_DAT_V2_INTEGERS = frozenset((
    'cycle_stamp', 'framecount', 'skip_frames', 'framelength', 'dturns',
    'skip_bins_start', 'skip_bins_end', 'min_dt_bin', 'max_dt_bin', 'rebin',
    'filmstart', 'filmstop', 'filmstep', 'niter', 'snpt', 'beam_ref_frame',
    'machine_ref_frame', 'c_time', 'ring'))
_DAT_V2_FLAGS = frozenset(('full_pp_flag', 'self_field_flag'))
# Version 3.0.0 frames field -> the version 2 value it holds. Every other
# version 2 value but raw_data_path is held by the parameters.
_DAT_V2_FRAMES_VALUES = {
    'nframes': 'framecount', 'nbins_frame': 'framelength',
    'skip_frames': 'skip_frames', 'skip_bins_start': 'skip_bins_start',
    'skip_bins_end': 'skip_bins_end', 'rebin': 'rebin',
    'sampling_time': 'dtbin'}


def detect_version(path: str) -> str:
    """Find the format version of an input file from its content.

    Parameters
    ----------
    path: str
        Path to the input file.

    Returns
    -------
    version: str
        The file's format version.

    Raises
    ------
    InputError: Exception
        Raised if the file names no format version.
    """
    return read_file(path)[1]


def read_file(path: str) -> tuple[str, str]:
    """Read the text of an input file and find its format version.

    Parameters
    ----------
    path: str
        Path to the input file. ``'stdin'`` or ``''`` reads from stdin: a
        JSON file to the end of the input, a `.dat` file to the end of its
        raw data.

    Returns
    -------
    text: str
        The file's text.
    version: str
        The file's format version.

    Raises
    ------
    InputError: Exception
        Raised if the file does not exist or names no format version.
    """
    text = ''.join(parsing.read_lines(path))
    version = detect_text_version(text)
    if version is None:
        raise expt.InputError(f'No format version found in "{path}"')
    return text, version


def detect_text_version(text: str) -> str | None:
    """Find the format version of the text of an input file.

    Parameters
    ----------
    text: str
        The text of the file.

    Returns
    -------
    version: str | None
        The format version, or None if the text names none.
    """
    if text.lstrip().startswith('{'):
        return json.loads(text).get('metadata', {}).get('format_version')

    # A `.dat` file names its version on the line after the preamble.
    lines = text.splitlines()
    version_line = (lines[parsing.DAT_FILE_HEADER_LENGTH]
                    if len(lines) > parsing.DAT_FILE_HEADER_LENGTH else '')
    return (version_line.removeprefix('! Version').strip()
            if version_line.startswith('! Version') else None)


def convert(input_path: str, output_path: str, version: str = JSON_VERSION):
    """Convert an input file of any supported format version to the given
    format version.

    Parameters
    ----------
    input_path: str
        Path to the file to convert.
    output_path: str
        Path of the file to write.
    version: str, optional, default=JSON_VERSION
        Format version to write, :data:`DAT_VERSION` for a `.dat` file.

    Raises
    ------
    InputError: Exception
        Raised if either format version is not supported.
    """
    text, input_version = read_file(input_path)
    text = convert_text(text, input_version, version)
    with open(output_path, 'w') as f:
        f.write(text)


def convert_text(text: str, from_version: str, to_version: str) -> str:
    """Convert the text of an input file between format versions, one
    version at a time.

    Parameters
    ----------
    text: str
        The text of the file.
    from_version: str
        Format version of the text.
    to_version: str
        Format version to convert to.

    Returns
    -------
    str
        The text of the file in the new format version.

    Raises
    ------
    InputError: Exception
        Raised if either format version is not supported.
    """
    start, end = _version_index(from_version), _version_index(to_version)
    step = 1 if end > start else -1
    for i in range(start, end, step):
        text = _STEPS[VERSIONS[i], VERSIONS[i + step]](text)
    return text


def format_json(document: dict) -> str:
    """Format a document of the latest JSON format version as text, with
    each frame of raw data on one line.

    Parameters
    ----------
    document: dict
        The document, as loaded from or to be written as JSON.

    Returns
    -------
    str
        The text of the JSON file.
    """
    rows = document['frames']['raw_data']
    text = json.dumps(
        {**document, 'frames': {**document['frames'], 'raw_data': None}},
        indent=2)

    # json.dumps cannot put each frame on one line, so the formatted rows
    # replace the placeholder, indented to sit inside the frames.
    if rows is not None:
        raw_data = ('[\n'
                    + ',\n'.join(' ' * 6 + json.dumps(row) for row in rows)
                    + '\n' + ' ' * 4 + ']')
        text = text.replace('"raw_data": null', f'"raw_data": {raw_data}')
    return text + '\n'


def dat_to_json(lines: Sequence[str],
                raw_data: NPArray | None = None) -> str:
    """Convert the parameter lines of a version 2 `.dat` file, and its raw
    data, to the text of a JSON file of the latest format version.

    Parameters
    ----------
    lines: Sequence[str]
        The `PARAMETER_LENGTH` lines before the raw data.
    raw_data: ndarray, optional, default=None
        The raw data, read from the file or the file it names.

    Returns
    -------
    str
        The text of the JSON file, before validation.

    Raises
    ------
    InputError: Exception
        Raised if the lines are not a `.dat` file's parameter lines, or a
        value cannot be read.
    """
    text = format_json(_dat_to_document_v3(lines, raw_data))
    return convert_text(text, '3.0.0', JSON_VERSION)


def _dat_to_document_v3(lines: Sequence[str],
                        raw_data: NPArray | None = None) -> dict:
    # The version 3.0.0 document for the parameter lines and raw data of a
    # version 2 `.dat` file.
    if not hasattr(lines, '__iter__'):
        raise expt.InputError('Input should be iterable')
    if len(lines) != parsing.PARAMETER_LENGTH:
        raise expt.InputError(
            f'Input array be of length {parsing.PARAMETER_LENGTH}, '
            f'containing every line of input file (not including '
            f'raw-data).')
    lines = [line.strip() for line in lines]

    parameters = {
        name: _from_dat_v2(name, lines[i])
        for name, i in parsing.DAT_FILE_LINE_INDEX.items()
        if name != 'raw_data_path'}
    frames = {field: parameters.pop(name)
              for field, name in _DAT_V2_FRAMES_VALUES.items()}

    preamble = lines[:parsing.DAT_FILE_HEADER_LENGTH]
    selector, _, context_name = preamble[0].partition('=')
    # The cycle stamp is given in seconds.
    stamp = _from_dat_v2('cycle_stamp', preamble[1])
    metadata = {
        'format_version': '3.0.0',
        'selector': selector or None,
        'cycle_stamp_ns': None if stamp is None else stamp * 10**9,
        'context_name': context_name or None,
        'acquisition_id': 1,
        'bunch_id': 0,
        'name': None,
        **{name: _from_dat_v2(name, text)
           for name, text in zip(parsing.DAT_FILE_PREAMBLE, preamble[2:])}}

    frames['raw_data_path'] = lines[parsing.RAW_DATA_FILE_IDX]
    frames['raw_data'] = None if raw_data is None else np.reshape(
        raw_data, (frames['nframes'], frames['nbins_frame'])).tolist()

    return {'metadata': metadata,
            'parameter_descriptions': pdesc.PARAMETER_DESCRIPTIONS_V3_0_0,
            'parameters': parameters,
            'frames': frames}


def _version_index(version: str) -> int:
    if version not in VERSIONS:
        raise expt.InputError(
            f'Unsupported format version {version!r}, supported versions '
            f'are {", ".join(VERSIONS)}')
    return VERSIONS.index(version)


def _from_dat_v2(name: str, text: str) -> str | int | float | bool | None:
    # Fortran index fields are 1-based in a `.dat` file and 0-based after.
    try:
        if name == 'output_dir':
            return text
        if text == '':
            return None
        if name in _DAT_V2_FLAGS:
            return {'0': False, '1': True}[text]
        if name in _DAT_V2_INTEGERS:
            value = int(text)
            return value - 1 if name in parsing.FORTRAN_TO_C_FIELDS else value
        return float(text)
    except (KeyError, ValueError) as exc:
        raise expt.InputError(
            f'Invalid legacy input:\n{name} is not valid: {text!r}') from exc


def _to_dat_v2(name: str, value: str | int | float | bool | None) -> str:
    # The inverse of _from_dat_v2, for a parameter.
    if value is None:
        if name == 'output_dir':
            raise expt.InputError(
                'output_dir has no value, which a .dat file cannot '
                'represent')
        # The format has no way to say "unset", so an optional parameter
        # without a value is written as zero.
        return '0'
    if isinstance(value, bool):
        return str(int(value))
    if name in parsing.FORTRAN_TO_C_FIELDS:
        return str(value + 1)
    return str(value)


def _dat_v2_to_json_v3(text: str) -> str:
    lines, raw_data = parsing.split_input(text.splitlines())
    return format_json(_dat_to_document_v3(lines, raw_data))


def _json_v3_to_dat_v2(text: str) -> str:
    document = json.loads(text)
    metadata = document.get('metadata', {})
    frames = document['frames']

    names = (metadata.get('selector') or '',
             metadata.get('context_name') or '')
    # The cycle stamp is written in seconds, so any fraction is lost.
    stamp = metadata.get('cycle_stamp_ns')
    preamble = ['='.join(names) if any(names) else None,
                None if stamp is None else stamp // 10**9,
                *(metadata.get(name) for name in parsing.DAT_FILE_PREAMBLE)]

    # An empty path names no external file, so the data goes inline.
    raw_data_path = frames.get('raw_data_path') or 'pipe'
    try:
        given = {**document['parameters'],
                 **{name: frames[field]
                    for field, name in _DAT_V2_FRAMES_VALUES.items()}}
        values = {name: _to_dat_v2(name, given[name])
                  for name in parsing.DAT_FILE_LINE_INDEX
                  if name != 'raw_data_path'}
    except KeyError as exc:
        raise expt.InputError(
            f'Version 3.0.0 input has no value for {exc}') from exc
    lines = (['' if value is None else str(value) for value in preamble]
             + parsing.DAT_FILE_TEMPLATE.format(
                 raw_data_path=raw_data_path, **values).splitlines())

    if raw_data_path == 'pipe':
        if frames.get('raw_data') is None:
            raise expt.RawDataImportError('No raw data held to write inline')
        lines += [str(value) for row in frames['raw_data'] for value in row]
    return '\n'.join(lines) + '\n'


#: (version, adjacent version) -> function converting the text of a file
#: from the first to the second. Supporting another version means adding it
#: to VERSIONS, with a step to and from its neighbour.
_STEPS = {
    ('2', '3.0.0'): _dat_v2_to_json_v3,
    ('3.0.0', '2'): _json_v3_to_dat_v2,
}
