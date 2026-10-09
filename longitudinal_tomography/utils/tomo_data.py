"""Pydantic models holding validated tomography input, with conversion to
and from legacy `.dat` files.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**, **Simon Albright**
"""
from __future__ import annotations
from typing import Annotated, TYPE_CHECKING

import numpy as np
import pydantic

from .. import assertions as asrt, exceptions as expt
from ..data import data_treatment as dtreat
from ..tomography import tomography
from ..tracking import particles as pts, tracking
from ..tracking.machine import Machine
from ..utils import (_input_parsing as parsing,
                     parameter_descriptions as pdesc,
                     tomo_config as conf)

if TYPE_CHECKING:
    from typing import Any, Collection, Sequence
    from numpy.typing import NDArray as NPArray

    from ..data.profiles import Profiles


class _ValidatedModel(pydantic.BaseModel):
    """Base model whose values can be checked after they are changed.

    Fields are validated when a model is created, but not when one is
    assigned to afterwards, so :meth:`validate` is the check to call on a
    model that has been edited.
    """

    # nested models are revalidated too, so checking a model checks
    # everything it holds
    model_config = pydantic.ConfigDict(revalidate_instances='always')

    def validate(self):
        """Raise if the model's current values are not valid.

        Raises
        ------
        ValidationError: Exception
            Raised if a value, or a combination of values, is invalid.
        """
        type(self).model_validate(dict(self))


class Parameters(_ValidatedModel):
    """Validated, type-coerced machine and reconstruction parameters.

    Field names match the canonical names in :data:`DAT_FILE_LINE_INDEX`.
    """

    output_dir: str | None = None
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
    filmstep: int = pydantic.Field(default=1, gt=0)
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
    g_coupling: float | None = None
    zwall_over_n: float | None = None
    pickup_sensitivity: float | None = None

    @pydantic.field_validator('charge')
    @classmethod
    def _charge_nonzero(cls, v: float) -> float:
        if v == 0:
            raise ValueError('charge must be nonzero')
        return v

    @pydantic.model_validator(mode='after')
    def _check_cross_field_bounds(self) -> Parameters:

        value_errors = []
        if not self.skip_frames < self.framecount:
            value_errors.append(
                f'skip_frames ({self.skip_frames}) must be '
                f'less than framecount ({self.framecount})')
        if not self.skip_bins_start < self.framelength:
            value_errors.append(
                f'skip_bins_start ({self.skip_bins_start}) must be less than '
                f'framelength ({self.framelength})')
        if not self.skip_bins_end < self.framelength:
            value_errors.append(
                f'skip_bins_end ({self.skip_bins_end}) must be less than '
                f'framelength ({self.framelength})')
        if not self.synch_part_x < self.framelength:
            value_errors.append(
                f'synch_part_x ({self.synch_part_x}) must be less than '
                f'framelength ({self.framelength})')
        if not self.filmstart < self.framecount:
            value_errors.append(
                f'filmstart ({self.filmstart}) must be less than '
                f'framecount ({self.framecount})')
        if not self.filmstop < self.framecount:
            value_errors.append(
                f'filmstop ({self.filmstop}) must be less than '
                f'framecount ({self.framecount})')
        if not self.beam_ref_frame < self.framecount:
            value_errors.append(
                f'beam_ref_frame ({self.beam_ref_frame}) must be less than '
                f'framecount ({self.framecount})')
        if not self.machine_ref_frame < self.framecount:
            value_errors.append(
                f'machine_ref_frame ({self.machine_ref_frame}) must be '
                f'less than framecount ({self.framecount})')

        if len(value_errors) != 0:
            merged_error = "Input values contain errors:"
            for e in value_errors:
                merged_error += f"\n  - {e}"
            raise ValueError(merged_error)

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
        if len(lines) != parsing.PARAMETER_LENGTH:
            raise expt.InputError(
                f'Input array be of length {parsing.PARAMETER_LENGTH}, '
                f'containing every line of input file (not including '
                f'raw-data).')

        stripped = [line.strip() for line in lines]

        param_values = {}
        for name, idx in parsing.DAT_FILE_LINE_INDEX.items():
            if name == 'raw_data_path':
                continue
            value = stripped[idx]
            if name in parsing.FORTRAN_TO_C_FIELDS:
                try:
                    value = int(value) - 1
                except ValueError as exc:
                    raise expt.InputError(
                        f'Invalid legacy input parameters:\n{name} on line '
                        f'{idx} is not an integer: {value!r}') from exc
            param_values[name] = value

        try:
            return cls(**param_values)
        except pydantic.ValidationError as exc:
            raise expt.InputError(
                f'Invalid legacy input parameters:\n{exc}') from exc

    def _dat_value(self, name: str) -> str:
        # Fortran index fields are stored 0-based, so the inverse of the
        # conversion in from_dat_lines is applied here.
        value = getattr(self, name)
        if value is None:
            if name == 'output_dir':
                raise expt.InputError(
                    'output_dir has no value, which a .dat file cannot '
                    'represent')
            # The format has no way to say "unset", so an optional
            # parameter without a value is written as zero.
            return '0'
        if isinstance(value, bool):
            return str(int(value))
        if name in parsing.FORTRAN_TO_C_FIELDS:
            return str(value + 1)
        return str(value)

    def to_dat_lines(self, raw_data_path: str = 'pipe',
                     preamble: Sequence[str] = ()) -> list[str]:
        """Render the parameters as the header lines of a legacy `.dat` file.

        Parameters
        ----------
        raw_data_path: str
            Value for the raw data path line, which is not a parameter.
            ``'pipe'`` means the raw data follows the header lines.
        preamble: Sequence[str], optional
            Lines preceding the parameters, padded with blank lines to
            `DAT_FILE_HEADER_LENGTH`.

        Returns
        -------
        list[str]
            The `PARAMETER_LENGTH` header lines, without line endings.

        Raises
        ------
        InputError: Exception
            Raised if a parameter holds no value the format can represent.
        """
        header = list(preamble)
        header += [''] * (parsing.DAT_FILE_HEADER_LENGTH - len(header))
        values = {name: self._dat_value(name)
                  for name in parsing.DAT_FILE_LINE_INDEX
                  if name != 'raw_data_path'}
        block = parsing.DAT_FILE_TEMPLATE.format(
            raw_data_path=raw_data_path, **values)
        return header + block.splitlines()


def _to_raw_data(value: Collection) -> NPArray:
    """Coerce raw data to a float array."""
    if not hasattr(value, '__iter__'):
        raise expt.RawDataImportError('Raw data should be iterable')
    return np.asarray(value, dtype=float)


#: Raw data as read from an input file, held as an array.
RawData = Annotated[
    pydantic.InstanceOf[np.ndarray],
    pydantic.BeforeValidator(_to_raw_data)]


class Frames(_ValidatedModel):
    """Class for storing raw data and information on how to treat them,
    based on a Fortran style input file.

    Arguments may be given positionally in the order below, or by keyword
    using either these names or the attribute names ``nframes``,
    ``nbins_frame`` and ``sampling_time``.

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
    raw_data: ndarray, None
        1D array holding raw data as a direct read from an input file.
    nprofs: int
        Number of profiles, i.e. frames after ``skip_frames`` is applied.
    nbins: int
        Number of bins in a profile, i.e. frame bins after
        ``skip_bins_start``/``skip_bins_end`` are applied.
    """

    # TODO: the aliases accept the legacy constructor argument names
    # framecount, framelength and dtbin alongside the field names, which
    # every other module already uses. Remove them, with the custom
    # __init__, at the next major version.
    nframes: int = pydantic.Field(
        gt=0, validation_alias=pydantic.AliasChoices('nframes', 'framecount'))
    nbins_frame: int = pydantic.Field(
        gt=0,
        validation_alias=pydantic.AliasChoices('nbins_frame', 'framelength'))
    skip_frames: int = pydantic.Field(ge=0)
    skip_bins_start: int = pydantic.Field(ge=0)
    skip_bins_end: int = pydantic.Field(ge=0)
    rebin: int = pydantic.Field(gt=0)
    sampling_time: float = pydantic.Field(
        gt=0, validation_alias=pydantic.AliasChoices('sampling_time', 'dtbin'))
    raw_data_path: str = ''
    raw_data: RawData | None = None

    def __init__(self, *args: Any, **kwargs: Any):
        # Positional arguments follow the legacy order; keyword arguments
        # accept either the legacy names or the field names. pydantic routes
        # every validation of a Frames through here, so both must be handled.
        legacy_order = ('framecount', 'framelength', 'skip_frames',
                        'skip_bins_start', 'skip_bins_end', 'rebin', 'dtbin',
                        'raw_data_path')
        if len(args) > len(legacy_order):
            raise TypeError(
                f'Frames takes at most {len(legacy_order)} positional '
                f'arguments but {len(args)} were given')
        super().__init__(**dict(zip(legacy_order, args)), **kwargs)

    def __eq__(self, other: object) -> bool:
        # pydantic's generated __eq__ compares the stored fields, where the
        # raw data array has no scalar truth value. Dumping it first gives
        # a list, which compares element-wise.
        if not isinstance(other, Frames):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    @pydantic.field_validator('raw_data')
    @classmethod
    def _flatten_raw_data(cls, raw_data: NPArray | None,
                          info: pydantic.ValidationInfo) -> NPArray | None:
        # Raw data given in 2D must hold one frame per row, and is held flat.
        if raw_data is not None and raw_data.ndim == 2:
            nframes = info.data.get('nframes')
            if raw_data.shape[0] != nframes:
                raise expt.RawDataImportError(
                    f'Raw data has {raw_data.shape[0]} rows, expected one '
                    f'per frame ({nframes})')
            raw_data = raw_data.ravel()
        return raw_data

    @pydantic.field_serializer('raw_data')
    def _dump_raw_data(self, raw_data: NPArray | None) -> list | None:
        # Dumped with one frame per row.
        if raw_data is None:
            return None
        return raw_data.reshape(self.nframes, self.nbins_frame).tolist()

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

    @pydantic.model_validator(mode='after')
    def _check_raw_data_length(self) -> Frames:
        if self.raw_data is not None:
            ndata = self.nframes * self.nbins_frame
            if len(self.raw_data) != ndata:
                raise expt.RawDataImportError(
                    f'Raw data has length {len(self.raw_data)}.\n'
                    f'expected length: {ndata}')
        return self

    @classmethod
    def from_parameters(cls, params: Parameters, raw_data_path: str = '',
                        raw_data: Collection | None = None) -> Frames:
        """Build Frames from the frame-shaping fields of a Parameters."""
        return cls(
            raw_data=raw_data,
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

    def to_waterfall(self, raw_data: NPArray | None = None) -> NPArray:
        """Function to convert from raw data to waterfall for use in
        reconstruction. The waterfall wil be shaped based on the
        settings found in the :class:`Frames` object.

        Shape of waterfall is generated based on parameters of the Frames
        object.

        Parameters
        ----------
        raw_data: ndarray, optional, default=None
            1D array holding all measured raw data. The raw data held by
            the object is used if none is given. The function works on a
            copy of the given raw_data.

        Returns
        -------
        waterfall: ndarray
            Raw-data shaped as waterfall (nprofiles, nbins).

        Raises
        ------
        RawDataImportError: Exception
            Raised if something is wrong with the raw data, or if none is
            given and the object holds none.
        """

        raw_data = self.raw_data if raw_data is None else raw_data

        if raw_data is None:
            raise expt.RawDataImportError(
                'No raw data given, and none is held')

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


class Metadata(_ValidatedModel):
    """Information describing input data and where it came from, held in
    full only by JSON files."""

    # Version of the file the data was read from, None if not read from one.
    format_version: str | None = None
    selector: str | None = None
    cycle_stamp_ns: int | None = None
    context_name: str | None = None
    acquisition_id: int = 1
    bunch_id: int = 0
    name: str | None = None
    c_time: int | None = None  # ms
    timespan: float | None = None  # µs
    ring: int | None = None
    delay: float | None = None  # ns
    v_scale: float | None = None  # V/div
    v_position: float | None = None  # div

    @classmethod
    def from_dat_lines(cls, lines: Sequence[str]) -> Metadata:
        """Build Metadata from the preamble of a legacy `.dat` file.

        Parameters
        ----------
        lines: Sequence[str]
            Lines of the `.dat` file, of which the first
            `DAT_FILE_HEADER_LENGTH` are read.

        Returns
        -------
        Metadata

        Raises
        ------
        InputError: Exception
            Raised if a preamble line holds an invalid value.
        """
        lines = [line.strip() for line in
                 lines[:parsing.DAT_FILE_HEADER_LENGTH]]
        selector, _, context_name = lines[0].partition('=')
        values = {'selector': selector, 'context_name': context_name,
                  **dict(zip(parsing.DAT_FILE_PREAMBLE, lines[2:]))}
        # pydantic's ValidationError is a ValueError too.
        try:
            # The cycle stamp is given in seconds.
            if lines[1]:
                values['cycle_stamp_ns'] = int(lines[1]) * 10**9
            # A blank line holds no value.
            return cls(**{name: value for name, value in values.items()
                          if value != ''})
        except ValueError as exc:
            raise expt.InputError(
                f'Invalid legacy input preamble:\n{exc}') from exc

    def to_dat_lines(self) -> list[str]:
        """Render the metadata as the preamble of a legacy `.dat` file.

        Returns
        -------
        list[str]
            The `DAT_FILE_HEADER_LENGTH` preamble lines, without line endings.
        """
        names = (self.selector or '', self.context_name or '')
        # The cycle stamp is written in seconds, so any fraction is lost.
        stamp = (None if self.cycle_stamp_ns is None
                 else self.cycle_stamp_ns // 10**9)
        values = ['='.join(names) if any(names) else None, stamp,
                  *(getattr(self, name) for name in parsing.DAT_FILE_PREAMBLE)]
        return ['' if value is None else str(value) for value in values]


class TomographyData(_ValidatedModel):
    """Full input data parsed from a legacy `.dat` file: the machine and
    reconstruction parameters plus the frame/raw-data metadata.
    """

    # Fields are dumped in this order, so the raw data in frames comes last.
    metadata: Metadata = pydantic.Field(default_factory=Metadata)
    parameter_descriptions: dict[str, str] = (
        pdesc.PARAMETER_DESCRIPTIONS_V3_0_0)
    parameters: Parameters
    frames: Frames

    @classmethod
    def from_dat_file(cls, path: str) -> TomographyData:
        """Build a TomographyData by reading and parsing a legacy `.dat` file.

        Parameters
        ----------
        path: str
            Path to the `.dat` file.

        Returns
        -------
        TomographyData
        """
        raw_params, raw_data = parsing.read_input(path)
        params = Parameters.from_dat_lines(raw_params)
        raw_data_path = raw_params[parsing.RAW_DATA_FILE_IDX]
        frame = Frames.from_parameters(params, raw_data_path=raw_data_path,
                                       raw_data=raw_data)
        return cls(parameters=params, frames=frame,
                   metadata=Metadata.from_dat_lines(raw_params))

    def to_dat_file(self, path: str):
        """Write the input data to a legacy `.dat` file.

        Raw data is appended to the parameter lines unless the frames name
        an external file holding it.

        Parameters
        ----------
        path: str
            Path of the file to write.

        Raises
        ------
        RawDataImportError: Exception
            Raised if raw data should be written inline but is missing.
        """
        # An empty path names no external file, so the data goes inline.
        raw_data_path = ('pipe' if self.frames.raw_data_path == ''
                          else self.frames.raw_data_path)

        lines = self.parameters.to_dat_lines(raw_data_path,
                                             self.metadata.to_dat_lines())
        if raw_data_path == 'pipe':
            if self.frames.raw_data is None:
                raise expt.RawDataImportError(
                    'No raw data held to write inline')
            lines += [str(value) for value in self.frames.raw_data.tolist()]

        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    def to_machine(self) -> Machine:
        """Build a Machine from the input data.

        Returns
        -------
        machine: Machine
            Machine object containing parameters for the machine and the
            tomographic reconstruction.
        """
        params = self.parameters

        # Calculating limits in phase for reconstruction area, specified by
        # user, from bin offsets to physical units of phase [s].
        min_dt = params.min_dt_bin * params.dtbin
        max_dt = (self.frames.nbins - params.max_dt_bin) * params.dtbin

        # Machine silently ignores unknown arguments, so the fields it does
        # not take are excluded explicitly.
        machine_kwargs = params.model_dump(exclude={
            'framecount', 'framelength', 'skip_frames', 'skip_bins_start',
            'skip_bins_end', 'min_dt_bin', 'max_dt_bin', 'rebin'})

        return Machine(**machine_kwargs, nprofiles=self.frames.nprofs,
                       nbins=self.frames.nbins, min_dt=min_dt, max_dt=max_dt)

    def to_profiles(self) -> Profiles:
        """Build Profiles from the held raw data, ready for tracking.

        A new Machine is built for the profiles. Its dtbin and synch_part_x
        are updated by the rebinning, and synch_part_x is fitted if the
        parameters ask for a fit.

        Returns
        -------
        profiles: Profiles
            Profiles object holding the treated waterfall, with the machine
            as ``profiles.machine``.

        Raises
        ------
        RawDataImportError: Exception
            Raised if no raw data is held.
        """
        machine = self.to_machine()
        profiles = parsing.raw_data_to_profiles(
            self.frames.to_waterfall(), machine, self.frames.rebin,
            self.frames.sampling_time)
        profiles.calc_profilecharge()

        if machine.synch_part_x < 0:
            fit_info = dtreat.fit_synch_part_x(profiles)
            machine.load_fitted_synch_part_x_ftn(fit_info)

        return profiles

    def run(self, reconstruct_profile: int | None = None) \
            -> tuple[NPArray, NPArray, NPArray]:
        """Perform a full reconstruction of one profile.

        Parameters
        ----------
        reconstruct_profile: int, optional, default=None
            Index of the profile to reconstruct. The parameters' filmstart
            is used if none is given.

        Returns
        -------
        t_range: ndarray
            1D array containing time axis of reconstructed phase space image.
        E_range: ndarray
            1D array containing energy axis of reconstructed phase space
            image.
        phase_space: ndarray
            2D array containing the reconstructed phase space image,
            normalised to sum to one.
        """
        profiles = self.to_profiles()
        machine = profiles.machine
        reconstr_idx = (machine.filmstart if reconstruct_profile is None
                        else reconstruct_profile)

        tracker = tracking.Tracking(machine)
        if machine.self_field_flag:
            profiles.calc_self_fields()
            tracker.enable_self_fields(profiles)

        xp, yp = tracker.track(reconstr_idx)

        # Converting from physical coordinates ([rad], [eV])
        # to phase space coordinates.
        if not tracker.self_field_flag:
            xp, yp = pts.physical_to_coords(
                xp, yp, machine, tracker.particles.xorigin,
                tracker.particles.dEbin)

        # Filters out lost particles, transposes particle matrix,
        # casts to np.int32.
        xp, yp = pts.ready_for_tomography(xp, yp, machine.nbins)

        tomo = tomography.Tomography(profiles.waterfall, xp, yp)
        tomo.run(niter=machine.niter)

        t_range, E_range, phase_space = dtreat.phase_space(
            tomo, machine, reconstr_idx)

        # Removing negative areas, then normalising.
        phase_space = phase_space.clip(0.0)
        phase_space /= conf.sum(phase_space)

        return t_range, E_range, phase_space
