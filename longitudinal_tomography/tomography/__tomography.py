"""Module containing the Tomography super class

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""

import logging

from abc import ABC, abstractmethod
import typing as t
import numpy as np

from .. import exceptions as expt

log = logging.getLogger(__name__)


class TomographyABC(ABC):
    """Base class for tomography classes.

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

    def __init__(self, waterfall: np.ndarray,
                 x_coords: np.ndarray = None, y_coords: np.ndarray = None):
        self._waterfall = self._normalize_profiles(waterfall.clip(0.0))

        self._nprofs: int = self.waterfall.shape[0]
        self._nbins: int = self.waterfall.shape[1]

        self.xp = x_coords
        self.yp = y_coords

        self.recreated = np.zeros(self.waterfall.shape)
        self.diff: np.ndarray = None
        self.weight: np.ndarray = None

    @property
    def waterfall(self) -> np.ndarray:
        """Waterfall defined as @property.

        Returns
        -------
        waterfall: ndarray
            Measured profiles (nprofiles, nbins).
            waterfall is normalized and its negative values are set to zero.
        """
        return self._waterfall

    @property
    def yp(self) -> np.ndarray:
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
    def yp(self, value: np.ndarray):
        if hasattr(value, '__iter__'):
            if self._xp is None:
                raise expt.CoordinateImportError(
                    'The object x-coordinates are None. x-coordinates'
                    'must be provided before the y-coordinates.')
            elif value.shape == self.xp.shape:
                self._yp = value.astype(np.int32)
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
    def xp(self) -> np.ndarray:
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
    def xp(self, value: np.ndarray):
        if hasattr(value, '__iter__'):
            value = np.array(value)
            if not value.ndim == 2:
                msg = 'X coordinates have two dimensions ' \
                      '(nparticles, nprofiles)'
                raise expt.CoordinateImportError(msg)
            if not value.shape[1] == self.nprofs:
                msg = f'Imported particles should be ' \
                      f'tracked trough {self.nprofs} profiles. ' \
                      f'Given particles seems so have been tracked trough ' \
                      f'{value.shape[1]} profiles.'
                raise expt.CoordinateImportError(msg)
            if np.any(value < 0) or np.any(value >= self.nbins):
                msg = 'X coordinate of particles outside of image width'
                raise expt.XPOutOfImageWidthError(msg)
            self._xp = np.ascontiguousarray(value).astype(np.int32)
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

    def _normalize_profiles(self, waterfall: np.ndarray) -> np.ndarray:
        if not waterfall.any():
            raise expt.WaterfallReducedToZero()
        waterfall /= np.sum(waterfall, axis=1)[:, None]
        return waterfall

    # Calculates discrepancy for the whole waterfall
    def _discrepancy(self, diff_waterfall: np.ndarray):
        return np.sqrt(
            np.sum(diff_waterfall ** 2) / (self.nbins * self.nprofs))

    # Created xp array modified to point at flattened version of waterfall.
    def _create_flat_points(self) -> np.ndarray:
        flat_points = self.xp.copy()
        for i in range(self.nprofs):
            flat_points[:, i] += self.nbins * i
        return flat_points

    # Finding the reciprocal of the number of particles in
    # a bin. Done to counterbalance the different amount of
    # particles in the different bins.
    # Bins with fewer particles have a larger amplification,
    # relative to bins containing many particles.
    def _reciprocal_particles(self) -> np.ndarray:
        ppb = np.zeros((self.nbins, self.nprofs))
        ppb = self._count_particles_in_bins(
            ppb, self.nprofs, self.xp, self.nparts)

        # Setting bins with zero particles one to avoid division by zero.
        ppb[ppb == 0] = 1
        return np.max(ppb) / ppb

    # Needed by reciprocal particles function.
    # TODO: removed njit, reimplement in C in the future
    def _count_particles_in_bins(self, ppb: np.ndarray,
                                 profile_count: int,
                                 xp: np.ndarray, nparts: int) -> np.ndarray:
        for i in range(profile_count):
            for j in range(nparts):
                ppb[xp[j, i], i] += 1
        return ppb

    @abstractmethod
    def run(self, niter: int = 20, verbose: bool = False,
            callback: t.Callable = None) -> np.ndarray:
        pass
