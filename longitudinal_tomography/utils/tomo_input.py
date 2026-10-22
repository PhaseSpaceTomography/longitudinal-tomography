"""Module containing functions for handling input from JSON and `.dat`
files of any format version.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING

import numpy as np
import pydantic

from .. import exceptions as expt
from ..utils import (_input_parsing as parsing, file_handling as fhand,
                     tomo_data as tdata)

if TYPE_CHECKING:
    from numpy.typing import NDArray as NPArray

    from ..data.profiles import Profiles
    from ..tracking.machine import Machine
    from ..tracking.machine_base import MachineABC

# Re-exported so that existing imports from this module keep working
Frames = tdata.Frames
PARAMETER_LENGTH = parsing.PARAMETER_LENGTH
RAW_DATA_FILE_IDX = parsing.RAW_DATA_FILE_IDX


def get_user_input(input: str = '') -> tuple[str, NPArray | None]:
    """Function to let user provide input as a JSON or `.dat` file either
    as file path or as stdin. Input of any format version is converted to
    JSON of the latest format version.

    The measured data can be held in the input file, or, for a `.dat` file,
    in a file it names.

    Returns
    -------
    tomo_input: tuple (str, ndarray)
        Tuple holding the input as JSON text, and ndarray containing the
        raw data it holds, or None if it holds none (parameters, raw data).
    """
    text, version = fhand.read_file(input)
    text = fhand.convert_text(text, version, fhand.JSON_VERSION)
    raw_data = json.loads(text)['frames'].get('raw_data')
    return text, (None if raw_data is None
                  else np.array(raw_data, dtype=float).ravel())


def txt_input_to_machine(input_array: str | list) \
        -> tuple[Machine, tdata.Frames]:
    """Function converts the content of an input file and uses this to
    generate an machine object. The input is given either as the text of
    an input file of any format version, such as the JSON text from
    :func:`get_user_input`, or as a list holding one line of a `.dat` file
    in each element, every line before the raw data included.

    Parameters
    ----------
    input_array: str | list
        The text of an input file, or the parameter lines of a `.dat` file.

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

    # TODO: `.dat` parameter lines hold no raw data, so frames from them
    # never do, unlike those from the text of a file, and callers have to
    # carry the array themselves. Drop the difference when the input seam
    # moves to a single reader.
    if isinstance(input_array, str):
        text = fhand.convert_text(input_array,
                                  fhand.detect_text_version(input_array),
                                  fhand.JSON_VERSION)
    else:
        text = fhand.dat_to_json(input_array)
    try:
        data = tdata.TomographyData.model_validate_json(text)
    except pydantic.ValidationError as exc:
        raise expt.InputError(f'Invalid input parameters:\n{exc}') from exc

    return data.to_machine(), data.frames



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
