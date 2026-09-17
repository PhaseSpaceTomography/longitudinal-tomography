"""Module containing functions for handling input from
Fortran style text files.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
from __future__ import annotations
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import pydantic

from .. import assertions as asrt, exceptions as expt
from ..data import pre_process
from ..data.profiles import Profiles
from ..tracking.machine import Machine
from ..compat import fortran

if TYPE_CHECKING:
    from typing import Collection, Sequence
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
OUTPUT_DIR_IDX = DAT_FILE_LINE_INDEX['output_dir']


class Parameters(pydantic.BaseModel):
    """Validated, type-coerced machine and reconstruction parameters.

    Field names match the canonical names in :data:`DAT_FILE_LINE_INDEX`.
    """

    output_dir: str = "/tmp/"
    framecount: int = pydantic.Field(gt=0)
    skip_frames: int = pydantic.Field(ge=0)
    framelength: int = pydantic.Field(gt=0)
    dtbin: float = pydantic.Field(gt=0)
    dturns: int = pydantic.Field(gt=0)
    skip_bins_start: int = pydantic.Field(ge=0)
    skip_bins_end: int = pydantic.Field(ge=0)
    min_dt_bin: int = pydantic.Field(ge=0)
    max_dt_bin: int = pydantic.Field(ge=0)
    rebin: int = pydantic.Field(gt=0)
    synch_part_x: float = -1.0
    demax: float = -1.0e6
    filmstart: int = pydantic.Field(default=0, ge=0)
    filmstop: int = pydantic.Field(default=1, ge=0)
    filmstep: int = pydantic.Field(gt=0)
    niter: int = pydantic.Field(default=20, gt=0)
    snpt: int = pydantic.Field(default=4, gt=0)
    full_pp_flag: bool = False
    beam_ref_frame: int = pydantic.Field(default=0, ge=0)
    machine_ref_frame: int = pydantic.Field(default=0, ge=0)
    vrf1: float
    vrf1dot: float = 0.0
    vrf2: float = 0.0
    vrf2dot: float = 0.0
    h_num: float = pydantic.Field(default=1.0, gt=0)
    h_ratio: float = pydantic.Field(default=1.0, gt=0)
    phi12: float = 0.0
    b0: float = pydantic.Field(gt=0)
    bdot: float = 0.0
    mean_orbit_rad: float = pydantic.Field(gt=0)
    bending_rad: float = pydantic.Field(gt=0)
    trans_gamma: float
    rest_energy: float = pydantic.Field(gt=0)
    charge: float = 1.0
    self_field_flag: bool = False
    g_coupling: float = 0
    zwall_over_n: float = 0
    pickup_sensitivity: float = 0

    @pydantic.field_validator('charge')
    @classmethod
    def _charge_nonzero(cls, v: float) -> float:
        if v == 0:
            raise ValueError('charge must be nonzero')
        return v

    @pydantic.model_validator(mode='after')
    def _check_cross_field_bounds(self) -> Parameters:
        if not self.skip_frames < self.framecount:
            raise ValueError(
                f'skip_frames ({self.skip_frames}) must be less than '
                f'framecount ({self.framecount})')
        if not self.skip_bins_start < self.framelength:
            raise ValueError(
                f'skip_bins_start ({self.skip_bins_start}) must be less '
                f'than framelength ({self.framelength})')
        if not self.skip_bins_end < self.framelength:
            raise ValueError(
                f'skip_bins_end ({self.skip_bins_end}) must be less than '
                f'framelength ({self.framelength})')
        if not self.synch_part_x < self.framelength:
            raise ValueError(
                f'synch_part_x ({self.synch_part_x}) must be less than '
                f'framelength ({self.framelength})')
        if not self.filmstart < self.framecount:
            raise ValueError(
                f'filmstart ({self.filmstart}) must be less than '
                f'framecount ({self.framecount})')
        if not self.filmstop < self.framecount:
            raise ValueError(
                f'filmstop ({self.filmstop}) must be less than '
                f'framecount ({self.framecount})')
        if not self.beam_ref_frame < self.framecount:
            raise ValueError(
                f'beam_ref_frame ({self.beam_ref_frame}) must be less than '
                f'framecount ({self.framecount})')
        if not self.machine_ref_frame < self.framecount:
            raise ValueError(
                f'machine_ref_frame ({self.machine_ref_frame}) must be '
                f'less than framecount ({self.framecount})')
        return self

    @classmethod
    def from_dat_lines(cls, lines: Sequence[str]) -> Parameters:
        """Build Parameters from the header lines of a legacy `.dat` file.

        Parameters
        ----------
        lines: Sequence[str]
            The `.dat` file's `PARAMETER_LENGTH` header lines (raw-data
            lines, if any, excluded).

        Returns
        -------
        Parameters
        """
        if not hasattr(lines, '__iter__'):
            raise expt.InputError('Input should be iterable')
        if len(lines) != PARAMETER_LENGTH:
            raise expt.InputError(
                f'Input array be of length {PARAMETER_LENGTH}, containing '
                f'every line of input file (not including raw-data).')

        stripped = [line.strip('\r\n') for line in lines]

        param_values = {}
        for name, idx in DAT_FILE_LINE_INDEX.items():
            if name == 'raw_data_path':
                continue
            value = stripped[idx]
            if name in FORTRAN_TO_C_FIELDS:
                value = int(value) - 1
            param_values[name] = value

        try:
            return cls(**param_values)
        except pydantic.ValidationError as exc:
            raise expt.InputError(
                f'Invalid legacy input parameters:\n{exc}') from exc


class Frames(pydantic.BaseModel):
    """Class for storing raw data and information on how to treat them,
    based on a Fortran style input file.

    Parameters
    ----------
    framecount: int
        Number of time frames.
    framelength: int
        Number of bins in a time frame.
    skip_frames: int
        Number of time frames to ignore from the
        beginning of the raw input file.
    skip_bins_start: int
        Subtract this number of bins from start of the raw input.
    skip_bins_end: int
        Subtract this number of bins from end of the raw input.
    rebin: int
        Rebinning factor - Number of frame bins to rebin into one profile bin.
    dtbin: float
        Size of profile bins (sampling size) [s].
    raw_data_path: string, optional, default=''
        Path to file holding raw data for programmers reference.

    Attributes
    ----------
    raw_data:
        Measured raw data as read from text file.
    nframes: int
        Number of measured time frames.
    nbins_frame: int
        Number of bins in a measured time frame.
    skip_frames: int
        Number of time frames to ignore from the
        beginning of the raw input file.
    skip_bins_start: int
        Subtract this number of bins from start of the raw input.
    skip_bins_end: int
        Subtract this number of bins from end of the raw input.
    rebin: int
        Rebinning factor, the number of frame bins to
        rebin into one profile bin.
    sampling_time: float
        Size of profile bins [s].
    raw_data_path: string
        Path to file holding raw data for programmers reference.
    nprofs: int
        Number of profiles, i.e. frames after ``skip_frames`` is applied.
    nbins: int
        Number of bins in a profile, i.e. frame bins after
        ``skip_bins_start``/``skip_bins_end`` are applied.
    """

    nframes: int = pydantic.Field(gt=0)
    nbins_frame: int = pydantic.Field(gt=0)
    skip_frames: int = pydantic.Field(ge=0)
    skip_bins_start: int = pydantic.Field(ge=0)
    skip_bins_end: int = pydantic.Field(ge=0)
    rebin: int = pydantic.Field(gt=0)
    sampling_time: float = pydantic.Field(gt=0)
    raw_data_path: str = ''

    _raw_data = pydantic.PrivateAttr(default=None)

    def __init__(self, framecount: int, framelength: int, skip_frames: int,
                 skip_bins_start: int, skip_bins_end: int, rebin: int,
                 dtbin: float, raw_data_path: str = ''):
        super().__init__(
            nframes=framecount, nbins_frame=framelength,
            skip_frames=skip_frames, skip_bins_start=skip_bins_start,
            skip_bins_end=skip_bins_end, rebin=rebin,
            sampling_time=dtbin, raw_data_path=raw_data_path)

    @pydantic.model_validator(mode='after')
    def _check_cross_field_bounds(self) -> Frames:
        if not self.skip_frames < self.nframes:
            raise ValueError(
                f'skip_frames ({self.skip_frames}) must be less than '
                f'nframes ({self.nframes})')
        if not self.skip_bins_start < self.nbins_frame:
            raise ValueError(
                f'skip_bins_start ({self.skip_bins_start}) must be less '
                f'than nbins_frame ({self.nbins_frame})')
        if not self.skip_bins_end < self.nbins_frame:
            raise ValueError(
                f'skip_bins_end ({self.skip_bins_end}) must be less than '
                f'nbins_frame ({self.nbins_frame})')
        return self

    @classmethod
    def from_parameters(cls, params: Parameters,
                         raw_data_path: str = '') -> Frames:
        """Build Frames from the frame-shaping fields of a Parameters."""
        return cls(
            framecount=params.framecount,
            framelength=params.framelength,
            skip_frames=params.skip_frames,
            skip_bins_start=params.skip_bins_start,
            skip_bins_end=params.skip_bins_end,
            rebin=params.rebin,
            dtbin=params.dtbin,
            raw_data_path=raw_data_path)

    @pydantic.computed_field
    @property
    def nprofs(self) -> int:
        """Number of profiles, to be used when creating a machine object."""
        return self.nframes - self.skip_frames

    @pydantic.computed_field
    @property
    def nbins(self) -> int:
        """Number of bins in a profile, to be used when creating a machine
        object."""
        return self.nbins_frame - self.skip_bins_start - self.skip_bins_end

    @property
    def raw_data(self) -> NPArray | None:
        """Raw data defined as a @property.

        Holds assertions for validity of raw data.

        Parameters
        ----------
        in_raw_data: ndarray
            1D array holding raw data as a direct read from text file.

        Returns
        -------
        raw_data: ndarray, none
            Copy of array holding raw data as a direct read from text file.
            None is returned if object holds no raw data.

        Raises
        ------
        RawDataImportError: Exception
            Raised if raw data is not iterable or has an unexpected length.

        """
        if self._raw_data is not None:
            return np.copy(self._raw_data)
        else:
            return None

    @raw_data.setter
    def raw_data(self, in_raw_data: Collection):
        if not hasattr(in_raw_data, '__iter__'):
            raise expt.RawDataImportError('Raw data should be iterable')

        ndata = self.nframes * self.nbins_frame
        if len(in_raw_data) == ndata:
            self._raw_data = np.array(in_raw_data)
        else:
            raise expt.RawDataImportError(
                f'Raw data has length {len(in_raw_data)}.\n'
                f'expected length: {ndata}')

    def to_waterfall(self, raw_data: NPArray) -> NPArray:
        """Function to convert from raw data to waterfall for use in
        reconstruction. The waterfall wil be shaped based on the
        settings found in the :class:`Frames` object.

        Shape of waterfall is generated based on parameters of the Frames
        object.

        Parameters
        ----------
        raw_data: ndarray
            1D array holding all measured raw data.
            The function works on a copy of the given raw_data.

        Returns
        -------
        waterfall: ndarray
            Raw-data shaped as waterfall (nprofiles, nbins).

        Raises
        ------
        RawDataImportError: Exception
            Raised if something is wrong with the raw data.
        """
        waterfall = self._assert_raw_data(raw_data)
        asrt.assert_frame_inputs(self)

        # Reshaping from raw data to waterfall
        waterfall = waterfall.reshape((self.nframes, self.nbins_frame))
        # Skips frames
        waterfall = waterfall[self.skip_frames:]

        # Skips bins at start and end of time frames.
        if self.skip_bins_end > 0:
            waterfall = waterfall[:, self.skip_bins_start:-self.skip_bins_end]
        else:
            waterfall = waterfall[:, self.skip_bins_start:]
        return waterfall

    # Check that provided raw data is valid.
    def _assert_raw_data(self, raw_data: NPArray | Collection) \
            -> NPArray:
        if not hasattr(raw_data, '__iter__'):
            raise expt.RawDataImportError('Raw data should be iterable')

        ndata = self.nframes * self.nbins_frame
        if not len(raw_data) == ndata:
            raise expt.RawDataImportError(
                f'Raw data has length {len(raw_data)}.\n'
                f'expected length: {ndata}')

        return np.array(raw_data)


class TomographyData(pydantic.BaseModel):
    """Full input data parsed from a legacy `.dat` file: the machine and
    reconstruction parameters plus the frame/raw-data metadata.
    """

    parameters: Parameters
    frames: Frames

    @classmethod
    def from_dat_file(cls, path: str) -> TomographyData:
        """Build a DatFile by reading and parsing a legacy `.dat` file.

        Parameters
        ----------
        path: str
            Path to the `.dat` file.

        Returns
        -------
        DatFile
        """
        raw_params, raw_data = get_user_input(path)
        params = Parameters.from_dat_lines(raw_params)
        raw_data_path = raw_params[DAT_FILE_LINE_INDEX['raw_data_path']]
        frame = Frames.from_parameters(params, raw_data_path=raw_data_path)
        frame.raw_data = raw_data
        return cls(parameters=params, frames=frame)


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
def _split_input(read_input: Sequence) -> tuple[list, NPArray]:
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


def txt_input_to_machine(input_array: list) -> tuple[Machine, Frames]:
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

    params = Parameters.from_dat_lines(input_array)
    raw_data_path = input_array[DAT_FILE_LINE_INDEX['raw_data_path']].strip(
        '\r\n')
    frame = Frames.from_parameters(params, raw_data_path=raw_data_path)

    # Calculating limits in phase for reconstruction area, specified by
    # user, from bin offsets to physical units of phase [s].
    min_dt = params.min_dt_bin * params.dtbin
    max_dt = (frame.nbins - params.max_dt_bin) * params.dtbin

    machine_kwargs = {
        'output_dir': params.output_dir,
        'dtbin': params.dtbin,
        'dturns': params.dturns,
        'synch_part_x': params.synch_part_x,
        'demax': params.demax,
        'filmstart': params.filmstart,
        'filmstop': params.filmstop,
        'filmstep': params.filmstep,
        'niter': params.niter,
        'snpt': params.snpt,
        'full_pp_flag': params.full_pp_flag,
        'beam_ref_frame': params.beam_ref_frame,
        'machine_ref_frame': params.machine_ref_frame,
        'vrf1': params.vrf1,
        'vrf1dot': params.vrf1dot,
        'vrf2': params.vrf2,
        'vrf2dot': params.vrf2dot,
        'h_num': params.h_num,
        'h_ratio': params.h_ratio,
        'phi12': params.phi12,
        'b0': params.b0,
        'bdot': params.bdot,
        'mean_orbit_rad': params.mean_orbit_rad,
        'bending_rad': params.bending_rad,
        'trans_gamma': params.trans_gamma,
        'rest_energy': params.rest_energy,
        'charge': params.charge,
        'self_field_flag': params.self_field_flag,
        'g_coupling': params.g_coupling,
        'zwall_over_n': params.zwall_over_n,
        'pickup_sensitivity': params.pickup_sensitivity,
        'nprofiles': frame.nprofs,
        'nbins': frame.nbins,
        'min_dt': min_dt,
        'max_dt': max_dt
    }

    # Creating machine object
    machine = Machine(**machine_kwargs)

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
