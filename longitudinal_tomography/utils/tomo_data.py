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
from ..utils import (_input_parsing as parsing, file_handling as fhand,
                     parameter_descriptions as pdesc,
                     tomo_config as conf)

if TYPE_CHECKING:
    from typing import Any, Collection
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
    The shape of the frames and their treatment are held by :class:`Frames`.
    """

    output_dir: str | None = None
    dturns: int = pydantic.Field(gt=0)
    min_dt_bin: int = pydantic.Field(ge=0)
    max_dt_bin: int = pydantic.Field(ge=0)
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
    pickup_sensitivity: float = pydantic.Field(default=1.0, gt=0, le=1)

    @pydantic.field_validator('charge')
    @classmethod
    def _charge_nonzero(cls, v: float) -> float:
        if v == 0:
            raise ValueError('charge must be nonzero')
        return v

    @pydantic.model_validator(mode='after')
    def _check_self_fields(self) -> Parameters:
        # The self-field calculation needs both.
        value_errors = []
        if self.self_field_flag:
            for name in ('g_coupling', 'zwall_over_n'):
                if getattr(self, name) is None:
                    value_errors.append(
                        f'{name} must be set when self_field_flag is true')

        _raise_value_errors(value_errors)
        return self


def _raise_value_errors(value_errors: list[str]):
    """Raise every failed check at once, if any failed."""
    if len(value_errors) != 0:
        merged_error = "Input values contain errors:"
        for e in value_errors:
            merged_error += f"\n  - {e}"
        raise ValueError(merged_error)


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
        # Bins are cut from both ends of a frame, and must leave at least
        # one.
        skip_bins = self.skip_bins_start + self.skip_bins_end
        if not skip_bins < self.nbins_frame:
            raise ValueError(
                f'skip_bins_start + skip_bins_end ({skip_bins}) must be less '
                f'than nbins_frame ({self.nbins_frame}):'
                f'    {self.skip_bins_start=}'
                f'    {self.skip_bins_end=}')
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

class TomographyData(_ValidatedModel):
    """Full input data, as held by a file of the latest JSON format version:
    the machine and reconstruction parameters, the frames holding the raw
    data, and the metadata.
    """

    # Fields are dumped in this order, so the raw data in frames comes last.
    metadata: Metadata = pydantic.Field(default_factory=Metadata)
    parameter_descriptions: dict[str, str] = (
        pdesc.PARAMETER_DESCRIPTIONS_V3_0_0)
    parameters: Parameters
    frames: Frames

    @pydantic.model_validator(mode='after')
    def _check_parameters_fit_frames(self) -> TomographyData:
        params, frames = self.parameters, self.frames

        value_errors = []
        if not params.synch_part_x < frames.nbins_frame:
            value_errors.append(
                f'synch_part_x ({params.synch_part_x}) must be less than '
                f'nbins_frame ({frames.nbins_frame})')
        # These index profiles, the frames left after skip_frames.
        for name in ('filmstart', 'filmstop', 'beam_ref_frame',
                     'machine_ref_frame'):
            if not getattr(params, name) < frames.nprofs:
                value_errors.append(
                    f'{name} ({getattr(params, name)}) must be less than the '
                    f'number of profiles, nframes - skip_frames '
                    f'({frames.nprofs})')

        _raise_value_errors(value_errors)
        return self

    @classmethod
    def from_file(cls, path: str) -> TomographyData:
        """Read input data from a file of any supported format version,
        converting it to the latest version first if needed.

        Parameters
        ----------
        path: str
            Path to the file. ``'stdin'`` or ``''`` reads from stdin.

        Returns
        -------
        TomographyData
            The input data, with the file's format version in its metadata.

        Raises
        ------
        InputError: Exception
            Raised if the file's format version is not supported, or the file
            holds invalid input.
        """
        text, version = fhand.read_file(path)
        try:
            data = cls.model_validate_json(
                fhand.convert_text(text, version, fhand.JSON_VERSION))
        except pydantic.ValidationError as exc:
            raise expt.InputError(
                f'Invalid input in "{path}":\n{exc}') from exc
        data.metadata.format_version = version
        return data

    def to_file(self, path: str, version: str = fhand.JSON_VERSION):
        """Write the input data to a file of the given format version,
        converting it from the latest version if needed.

        Parameters
        ----------
        path: str
            Path of the file to write.
        version: str, optional, default=JSON_VERSION
            Format version to write, :data:`DAT_VERSION` for a `.dat` file.

        Raises
        ------
        InputError: Exception
            Raised if the format version is not supported, or the data
            cannot be represented in it.
        RawDataImportError: Exception
            Raised if a `.dat` file should hold raw data that is missing.
        """
        metadata = self.metadata.model_copy(
            update={'format_version': fhand.JSON_VERSION})
        # The profile counts are derived, so they are left out and the raw
        # data ends the file.
        document = self.model_copy(update={'metadata': metadata}).model_dump(
            mode='json', exclude={'frames': {'nprofs', 'nbins'}})
        text = fhand.convert_text(fhand.format_json(document),
                                  fhand.JSON_VERSION, version)
        with open(path, 'w') as f:
            f.write(text)

    def to_machine(self) -> Machine:
        """Build a Machine from the input data.

        Returns
        -------
        machine: Machine
            Machine object containing parameters for the machine and the
            tomographic reconstruction.
        """
        params, frames = self.parameters, self.frames

        # Calculating limits in phase for reconstruction area, specified by
        # user, from bin offsets to physical units of phase [s].
        min_dt = params.min_dt_bin * frames.sampling_time
        max_dt = (frames.nbins - params.max_dt_bin) * frames.sampling_time

        # Machine silently ignores unknown arguments, so the fields it does
        # not take are excluded explicitly.
        machine_kwargs = params.model_dump(exclude={'min_dt_bin',
                                                    'max_dt_bin'})

        return Machine(**machine_kwargs, dtbin=frames.sampling_time,
                       nprofiles=frames.nprofs, nbins=frames.nbins,
                       min_dt=min_dt, max_dt=max_dt)

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
