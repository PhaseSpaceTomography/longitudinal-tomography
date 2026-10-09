"""Module containing functions for handling input from
Fortran style text files.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from ..utils import _input_parsing as parsing, tomo_data as tdata

if TYPE_CHECKING:
    from numpy.typing import NDArray as NPArray

    from ..data.profiles import Profiles
    from ..tracking.machine import Machine
    from ..tracking.machine_base import MachineABC

# Re-exported so that existing imports from this module keep working
Frames = tdata.Frames
PARAMETER_LENGTH = parsing.PARAMETER_LENGTH
RAW_DATA_FILE_IDX = parsing.RAW_DATA_FILE_IDX


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
    return parsing.read_input(input)


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
    return parsing.raw_data_to_profiles(waterfall, machine, rbn,
                                        sampling_time, synch_part_x)
