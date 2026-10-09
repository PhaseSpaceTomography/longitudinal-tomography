"""Module containing functions for handling input from
Fortran style text files.

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
from ..utils import tomo_data as tdata

if TYPE_CHECKING:
    from typing import Sequence
    from numpy.typing import NDArray as NPArray

    from ..tracking.machine import Machine
    from ..tracking.machine_base import MachineABC

# Re-exported from tomo_data
Frames = tdata.Frames
PARAMETER_LENGTH = tdata.PARAMETER_LENGTH

RAW_DATA_FILE_IDX = tdata.DAT_FILE_LINE_INDEX['raw_data_path']


def get_user_input(input: str = ''):
    """Function to let user provide input as a text file either as
    file path or as stdin.

    If a filepath is given through the **system arguments**, the path to the
    output directory can be given as the second argument. The specified
    output path will substitute the output path read from the input file.
    A failed path to the measured data, or the measured data itself can
    be given in the input file.

    If the input is provided via **stdin**, the measured data must be
    stored in the same input file.

    Returns
    -------
    tomo_input: tuple (list, ndarray)
        Tuple holding machine and reconstruction parameters as a list of
        strings directly read from text file and ndarray containing
        raw data (parameters, raw data).
    """
    if input == 'stdin' or input == '':
        read = _get_input_stdin()
    else:
        read = _get_input_args(input)
    return _split_input(read)


# Receive path to input file via sys.argv.
# Can also receive the path to the output directory.
def _get_input_args(input_file_pth: str) -> NPArray:
    if not os.path.isfile(input_file_pth):
        raise expt.InputError(f'The input file: "{input_file_pth}" '
                              f'does not exist!')

    with open(input_file_pth, 'r') as f:
        read = f.readlines()

    return np.array(read)


# Read machine parameters via stdin.
# Here the measured data must be pipelined in the same file as
# the machine parameters.
def _get_input_stdin() -> NPArray:
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
            if line_num == tdata.DAT_FILE_LINE_INDEX['framecount']:
                nframes = int(read[-1])
            if line_num == tdata.DAT_FILE_LINE_INDEX['framelength']:
                nbins = int(read[-1])
                ndata_points += nframes * nbins
        if line_num == ndata_points:
            finished = True
        line_num += 1

    return np.array(read)


# Splits the read input data to machine parameters and raw data.
# If the raw data is not already read from the input file, the
#  data will be found in the file given by the parameter file.
def _split_input(read_input: Sequence) -> tuple[list, NPArray]:
    nframes_idx = tdata.DAT_FILE_LINE_INDEX['framecount']
    nbins_idx = tdata.DAT_FILE_LINE_INDEX['framelength']
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


def txt_input_to_machine(input_array: list) -> tuple[Machine, tdata.Frames]:
    """Function converts the content of an input file and uses this to
    generate an machine object. The input file is given as a list
    holding one line of the file in ach element. The list should
    contain a direct read from a Fortran styled input file, where all
    lines should be included.

    Parameters
    ----------
    input_array: list
        Array containing each line of an input file as an element of the array.
        This should be a direct read from a tomography text file.

    Returns
    -------
    machine: Machine
        Machine object containing parameters for the machine and the
        tomographic reconstruction.
    frame: Frames
        TODO: A frame object

    Raises
    ------
    InputError: Exception
        Error in input array
    """

    params = tdata.Parameters.from_dat_lines(input_array)
    raw_data_path = input_array[RAW_DATA_FILE_IDX].strip()
    # TODO: only the parameter lines are given here, so these frames never
    # hold raw data, unlike those built by TomographyData.from_dat_file, and
    # callers have to carry the array themselves. Drop the difference when
    # the input seam moves to a single reader.
    frame = tdata.Frames.from_parameters(params, raw_data_path=raw_data_path)
    machine = tdata.TomographyData(
        parameters=params, frames=frame).to_machine()

    return machine, frame


def raw_data_to_profiles(waterfall: NPArray, machine: MachineABC, rbn: int,
                         sampling_time: float,
                         synch_part_x: float = None) -> Profiles:
    """Converts from waterfall of untreated data, to waterfall
    ready for for tomography. The input waterfall is copied, and the
    treated waterfall is saved to a profiles object.

    The data treatment in this function is the same as in the original
    original tomography program:

    - Subtracts baseline from measurements
    - Re-binned profiles
    - Creates Profiles object to store waterfall
    - Negative values of waterfall is set to zero.

    **NB: dtbin and synch_part_x of the provided machine object
    will be updated after re-binning.**

    Parameters
    ----------
    waterfall: ndarray
        Raw-data shaped as waterfall (nprofiles, nbins).
    machine: Machine
        Machine object holding machine and reconstruction parameters.
    rbn: int
        Rebinning factor, the number of frame bins to
        rebin into one profile bin.
    sampling_time: float
        Size of profile bins [s].
    synch_part_x: float, optional, default=None
        X-coordinate of synchronous particle. If not given as argument,
        the machine.synch_part_x will be used.

    Returns
    -------
    profile: Profiles
        Profiles object holding the waterfall and information about the
        measurements.
    """
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
