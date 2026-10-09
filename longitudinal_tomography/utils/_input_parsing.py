"""Parsing of tomography input from Fortran style text files, shared by
:mod:`tomo_input` and :mod:`tomo_data`.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
from __future__ import annotations
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

from .. import exceptions as expt
from ..data import pre_process
from ..data.profiles import Profiles
from ..compat import fortran

if TYPE_CHECKING:
    from typing import Sequence
    from numpy.typing import NDArray as NPArray

    from ..tracking.machine_base import MachineABC

PARAMETER_LENGTH = 98

#: Canonical parameter name -> 0-based line index in a `.dat` file.
DAT_FILE_LINE_INDEX: dict[str, int] = {
    'raw_data_path': 12,
    'output_dir': 14,
    'framecount': 16,
    'skip_frames': 18,
    'framelength': 20,
    'dtbin': 22,
    'dturns': 24,
    'skip_bins_start': 26,
    'skip_bins_end': 28,
    'min_dt_bin': 31,
    'max_dt_bin': 34,
    'rebin': 36,
    'synch_part_x': 39,
    'demax': 41,
    'filmstart': 43,
    'filmstop': 45,
    'filmstep': 47,
    'niter': 49,
    'snpt': 51,
    'full_pp_flag': 53,
    'beam_ref_frame': 55,
    'machine_ref_frame': 57,
    'vrf1': 61,
    'vrf1dot': 63,
    'vrf2': 65,
    'vrf2dot': 67,
    'h_num': 69,
    'h_ratio': 71,
    'phi12': 73,
    'b0': 75,
    'bdot': 77,
    'mean_orbit_rad': 79,
    'bending_rad': 81,
    'trans_gamma': 83,
    'rest_energy': 85,
    'charge': 87,
    'self_field_flag': 91,
    'g_coupling': 93,
    'zwall_over_n': 95,
    'pickup_sensitivity': 97,
}
#: `.dat` fields that are 1-based in the file and must be converted to
#: 0-based indices once, when building a :class:`Parameters` from a `.dat`
#: file's lines.
FORTRAN_TO_C_FIELDS = frozenset((
    'filmstart', 'filmstop', 'beam_ref_frame', 'machine_ref_frame'))
RAW_DATA_FILE_IDX = DAT_FILE_LINE_INDEX['raw_data_path']

#: Number of lines preceding the parameter block of a `.dat` file.
DAT_FILE_HEADER_LENGTH = 8
#: Metadata field held on each of lines 2 to 7 of a `.dat` file, in order.
#: Line 0 holds ``selector=context_name`` and line 1 the cycle stamp in
#: seconds.
DAT_FILE_PREAMBLE = ('c_time', 'timespan', 'ring', 'delay', 'v_scale',
                     'v_position')

#: The parameter block of a `.dat` file, lines 8 to 97, with a placeholder
#: at every line that holds a value. The one spliced comment line is only
#: split to stay within the line length limit, and the escaped space
#: preserves a trailing space the Fortran writer emits.
DAT_FILE_TEMPLATE = (
    """! Version 2
!
! Data Format:
! Input data file =
{raw_data_path}
! Directory in which to write all output =
{output_dir}
! Number of frames in input data =
{framecount}
! Number of frames to ignore =
{skip_frames}
! Number of bins in each frame =
{framelength}
! Width (in s) of each frame bin =
{dtbin}
! Number of machine turns between frames =
{dturns}
! Number of frame bins before the lower profile bound to ignore =
{skip_bins_start}
! Number of frame bins after the upper profile bound to ignore =
{skip_bins_end}
! Number of frame bins after the lower profile bound to treat as empty
! at the reconstructed time =
{min_dt_bin}
! Number of frame bins before the upper profile bound to treat as empty
! at the reconstructed time =
{max_dt_bin}
! Number of frame bins to rebin into one profile bin =
{rebin}
! Time (in frame bins) from the lower profile bound to the synchronous phase
! (if <0, a fit is performed) in the "bunch reference" frame =
{synch_part_x}
! Max energy (in eV) of reconstructed phase space (if >0) =
{demax}
! Number of the first profile at which to reconstruct =
{filmstart}
! Number of the last profile at which to reconstruct =
{filmstop}
! Step between reconstructions =
{filmstep}
! Number of iterations for each reconstruction =
{niter}
! Square root of the number of test particles to track per cell =
{snpt}
! Flag to extend the region in phase space of map elements (if =1) =
{full_pp_flag}
"""
    '! Reference frame for bunch parameters (synchronous phase, '
    'baseline, integral) =\n'
    """{beam_ref_frame}
! Reference frame for machine parameters (RF voltages, B-field) =
{machine_ref_frame}
!
! Machine and Particle Parameters:
! Peak RF voltage (in V) of principal RF system =
{vrf1}
! and its time derivative (in V/s) =
{vrf1dot}
! Peak RF voltage (in V) of higher-harmonic RF system =
{vrf2}
! and its time derivative (in V/s) =
{vrf2dot}
! Harmonic number of principal RF system =
{h_num}
! Ratio of harmonics between RF systems =
{h_ratio}
! Phase difference (in radians of the principal harmonic) between RF systems =
{phi12}
! Dipole magnetic field (in T) =
{b0}
! and its time derivative (in T/s) =
{bdot}
! Machine radius (in m) =
{mean_orbit_rad}
! Bending radius (in m) =
{bending_rad}
! Gamma transition =
{trans_gamma}
! Rest mass (in eV/c**2) of accelerated particle =
{rest_energy}
! Charge state of accelerated particle =
{charge}
!
! Space Charge Parameters:
! Flag to include self-fields in the tracking (if =1) =
{self_field_flag}
! Geometrical coupling coefficient =\x20
{g_coupling}
! Reactive impedance (in Ohms per mode number) over a machine turn =
{zwall_over_n}
! Effective pick-up sensitivity (in digitizer units per instantaneous Amp) =
{pickup_sensitivity}"""
)


def read_input(input: str = '') -> tuple[list, NPArray]:
    """Read input from stdin, or from the file at the given path, and
    split it into parameter lines and raw data."""
    return split_input(read_lines(input))


def read_lines(input: str = '') -> NPArray:
    """Read every line of input from stdin, or from the file at the given
    path."""
    if input == 'stdin' or input == '':
        return get_input_stdin()
    return get_input_args(input)


# Receive path to input file via sys.argv.
# Can also receive the path to the output directory.
def get_input_args(input_file_pth: str) -> NPArray:
    if not os.path.isfile(input_file_pth):
        raise expt.InputError(f'The input file: "{input_file_pth}" '
                              f'does not exist!')

    with open(input_file_pth, 'r') as f:
        read = f.readlines()

    return np.array(read)


# Read machine parameters via stdin.
# Here the measured data must be pipelined in the same file as
# the machine parameters.
def get_input_stdin() -> NPArray:
    read = []
    finished = False
    piped_raw_data = False

    line_num = 0
    ndata_points = PARAMETER_LENGTH

    while line_num < ndata_points:
        read.append(sys.stdin.readline())
        if line_num == RAW_DATA_FILE_IDX:
            if 'pipe' in read[-1]:
                piped_raw_data = True
        if piped_raw_data:
            if line_num == DAT_FILE_LINE_INDEX['framecount']:
                nframes = int(read[-1])
            if line_num == DAT_FILE_LINE_INDEX['framelength']:
                nbins = int(read[-1])
                ndata_points += nframes * nbins
        if line_num == ndata_points:
            finished = True
        line_num += 1

    return np.array(read)


# Splits the read input data to machine parameters and raw data.
# If the raw data is not already read from the input file, the
#  data will be found in the file given by the parameter file.
def split_input(read_input: Sequence) -> tuple[list, NPArray]:
    nframes_idx = DAT_FILE_LINE_INDEX['framecount']
    nbins_idx = DAT_FILE_LINE_INDEX['framelength']
    ndata = 0
    read_parameters = None
    read_data = None

    try:
        read_parameters = read_input[:PARAMETER_LENGTH]
        ndata = (int(read_parameters[nbins_idx])
                 * int(read_parameters[nframes_idx]))
        for i in range(PARAMETER_LENGTH):
            read_parameters[i] = read_parameters[i].strip('\r\n')
    except TypeError:
        err_msg = 'Something went wrong while accessing machine parameters.'
        raise expt.InputError(err_msg)

    if read_parameters[RAW_DATA_FILE_IDX] == 'pipe':
        try:
            read_data = np.array(read_input[PARAMETER_LENGTH:], dtype=float)
        except TypeError:
            err_msg = 'Pipelined raw-data could not be casted to float.'
            raise expt.InputError(err_msg)
    else:
        try:
            read_data = np.genfromtxt(read_parameters[RAW_DATA_FILE_IDX],
                                      dtype=float)
        except FileNotFoundError:
            err_msg = f'The given file path for the raw-data:\n' \
                      f'{read_parameters[RAW_DATA_FILE_IDX]}\n' \
                      f'Could not be found'
            raise FileNotFoundError(err_msg)
        except Exception:
            err_msg = 'Something went wrong while loading raw_data.'
            raise Exception(err_msg)

    if not len(read_data) == ndata:
        raise expt.InputError(f'Wrong amount of data points loaded.\n'
                              f'Expected: {ndata}\n'
                              f'Loaded:   {len(read_data)}')

    return read_parameters, read_data


def raw_data_to_profiles(waterfall: NPArray, machine: MachineABC,
                         rbn: int, sampling_time: float,
                         synch_part_x: float = None) -> Profiles:
    """Subtract the baseline from a waterfall, rebin it, and store it
    in a Profiles object. The machine's dtbin and synch_part_x are
    updated by the rebinning."""
    if not hasattr(waterfall, '__iter__'):
        raise expt.WaterfallError('Waterfall should be an iterable')
    if synch_part_x is None:
        synch_part_x = machine.synch_part_x

    waterfall = np.array(waterfall)
    # Subtracting baseline
    waterfall[:] -= fortran.calc_baseline(waterfall, machine.beam_ref_frame)
    # Rebinning
    (waterfall,
     machine.dtbin,
     machine.synch_part_x) = pre_process.rebin(
        waterfall, rbn, sampling_time, synch_part_x)
    # Returning Profiles object.
    return Profiles(machine, sampling_time, waterfall)
