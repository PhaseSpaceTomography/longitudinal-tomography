"""Module containing the Tomography super class with

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""

import logging

from abc import ABC, abstractmethod
import typing as t
import cupy as cp
from ..utils.execution_mode import Mode

from .. import exceptions as expt

log = logging.getLogger(__name__)


class TomographyCuPyABC(ABC):
    """Base class for tomography classes using CuPy.

    This class holds tomography utilities and assertions.

    Parameters
    ----------
    waterfall: ndarray
        2D array of measured profiles, shaped: (nprofiles, nbins).
    x_coords: ndarray: int
        x-coordinates of particles, given as coordinates of the reconstructed
        phase space coordinate system. Shape: (nparts, nprofiles).

    Attributes
    ----------
    nparts: int
        Number of test particles.
    nprofs: int
        Number of profiles (time frames).
    nbins: int
        Number of bins in each profile.
    waterfall: ndarray
        2D array of measured profiles, shaped: (nprofiles, nbins).
        Negative values of waterfall is set to zero, and the waterfall is
        normalized.
    xp: ndarray
        x-coordinates of particles, given as coordinates of the reconstructed
        phase space coordinate system. Shape: (nparts, nprofiles).
    recreated: ndarray
        Recreated waterfall. Directly comparable with
        *Tomography.waterfall*. Shape: (nprofiles, nbins).
    diff: ndarray
        Discrepancy for phase space reconstruction at each iteration
        of the reconstruction process.
    """

    def __init__(self, waterfall: cp.ndarray,
                 x_coords: cp.ndarray = None, y_coords: cp.ndarray = None):
        self._waterfall = self._normalize_profiles(waterfall.clip(0.0))

        self._nprofs: int = self.waterfall.shape[0]
        self._nbins: int = self.waterfall.shape[1]

        self.xp = x_coords
        self.yp = y_coords

        self.recreated = cp.zeros(self.waterfall.shape)
        self.diff: cp.ndarray = None
        self.weight: cp.ndarray = None

    @property
    def waterfall(self) -> cp.ndarray:
        """Waterfall defined as @property.

        Returns
        -------
        waterfall: ndarray
            Measured profiles (nprofiles, nbins).
            waterfall is normalized and its negative values are set to zero.
        """
        return self._waterfall

    @property
    def yp(self) -> cp.ndarray:
        """Y-coordinates defined as @property.

        Parameters
        ----------
        value: ndarray, None
            2D array of tracked particles energy coordinates,
            given in phase space coordinates as integers.
            Shape: (nparts, nprofiles).

            By setting tomography.yp = None, the saved y-coordinates
            are deleted.

        Returns
        -------
        xp: ndarray
            particles y-coordinates, given as coordinates of the reconstructed
            phase space coordinate system in integers.
            shape: (nparts, nprofiles).

        Raises
        ------
        CoordinateImportError: Exception
            Provided coordinates are invalid.
        """
        return self._yp

    @yp.setter
    def yp(self, value: cp.ndarray):
        if hasattr(value, '__iter__'):
            if self._xp is None:
                raise expt.CoordinateImportError(
                    'The object x-coordinates are None. x-coordinates'
                    'must be provided before the y-coordinates.')
            elif value.shape == self.xp.shape:
                self._yp = value.astype(cp.int32)
            else:
                raise expt.CoordinateImportError(
                    'The given y-coordinates should be of the '
                    'same shape as the x-coordinates.')
        elif value is None:
            self._yp = None
        else:
            raise expt.CoordinateImportError(
                'Y-coordinates should be iterable, or None.')

    @property
    def xp(self) -> cp.ndarray:
        """X-coordinates defined as @property.

        Automatically updates `nparts`.

        Parameters
        ----------
        value: ndarray, None
            2D array of tracked particles phase coordinates,
            given in phase space coordinates as integers.
            Shape: (nparts, nprofiles).

            By setting tomography.xp = None, the saved x-coordinates
            are deleted.

        Returns
        -------
        xp: ndarray
            particles x-coordinates, given as coordinates of the reconstructed
            phase space coordinate system in integers.
            shape: (nparts, nprofiles).

        Raises
        ------
        CoordinateImportError: Exception
            Provided coordinates are invalid.
        XPOutOfImageWidthError: Exception
            Particle(s) out of image width.
        """
        return self._xp

    @xp.setter
    def xp(self, value: cp.ndarray):
        if hasattr(value, '__iter__'):
            value = cp.asarray(value)
            if value.ndim != 2:
                msg = 'X coordinates have two dimensions ' \
                      '(nparticles, nprofiles)'
                raise expt.CoordinateImportError(msg)
            if not value.shape[1] == self.nprofs:
                msg = f'Imported particles should be ' \
                      f'tracked trough {self.nprofs} profiles. ' \
                      f'Given particles seems so have been tracked trough ' \
                      f'{value.shape[1]} profiles.'
                raise expt.CoordinateImportError(msg)
            if cp.any(cp.logical_or(value < 0, value >= self.nbins)):
                msg = 'X coordinate of particles outside of image width'
                raise expt.XPOutOfImageWidthError(msg)
            self._xp = cp.ascontiguousarray(value, dtype=cp.int32)
            self._nparts = self._xp.shape[0]
            log.info(f'X coordinates of shape {self.xp.shape} loaded.')
        elif value is None:
            self._xp = None
            self._nparts = None
            log.info('X coordinates set to None')
        else:
            msg = 'X coordinates should be iterable, or None.'
            raise expt.CoordinateImportError(msg)

    @property
    def nparts(self) -> int:
        """Number of particles defined as @property.

        Returns
        -------
        nparts: int
            Number of test particles.
        """
        return self._nparts

    @property
    def nbins(self) -> int:
        """Number of profile bins defined as @property.

        Returns
        -------
        nbins: int
            Number of bins in each profile.
        """
        return self._nbins

    @property
    def nprofs(self) -> int:
        """Number of profiles defined as @property.

        nprofs: int
            Number of profiles (time frames).
        """
        return self._nprofs

    def _normalize_profiles(self, waterfall: cp.ndarray) -> cp.ndarray:
        if not waterfall.any():
            raise expt.WaterfallReducedToZero()
        waterfall /= cp.sum(waterfall, axis=1)[:, None]
        return waterfall

    @abstractmethod
    def run(self, niter: int = 20, verbose: bool = False, mode: Mode = Mode.CUDA) -> cp.ndarray:
        pass
