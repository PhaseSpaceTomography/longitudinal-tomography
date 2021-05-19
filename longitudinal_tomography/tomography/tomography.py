"""Module containing the TomographyCpp class

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""

import logging
import typing as t

import numpy as np

from .__tomography import TomographyABC
from ..cpp_routines import libtomo
from .. import exceptions as expt

log = logging.getLogger(__name__)


class Tomography(TomographyABC):
    """Class for performing tomographic reconstruction of phase space.

    The tomographic routine largely consists of two parts. Projection and
    back projection. The **back projection** creates a phase space
    reconstruction based on the measured profiles. The **projection**
    routine converts from back projection to reconstructed profiles.

    By comparing the reconstructed profiles to the measured profiles,
    adjustments of the weights can be made in order to create
    a better reconstruction. The number of iterations in this process can be
    specified by the user.

    Parameters
    ----------
    waterfall: ndarray
        2D array of measured profiles, shaped: (nprofiles, nbins).
    x_coords: ndarray
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
        Recreated waterfall. Directly comparable with *Tomography.waterfall*.
        Shape: (nprofiles, nbins).
    diff: ndarray
        Discrepancy for phase space reconstruction at each iteration
        of the reconstruction process.
    """

    def __init__(self, waterfall: np.ndarray, x_coords: np.ndarray = None,
                 y_coords: np.ndarray = None):
        super().__init__(waterfall, x_coords, y_coords)

    def run_hybrid(self, niter=20, verbose=False):
        """Function to perform tomographic reconstruction, implemented
        as a hybrid between C++ and Python.

        Projection and back projection routines are called from C++,
        the rest is written in python. Kept for reference.

        The discrepancy of each iteration is saved in the *diff* array
        of the object.

        The recreated waterfall can be found calling the *recreated* field
        of the object.

        Parameters
        ----------
        niter: int
            Number of iterations in reconstruction.
        verbose: boolean
            Flag to indicate that the status of the tomography should be
            written to stdout. The output is identical to output
            generated in the original Fortran tomography.

        Returns
        -------
        weight: ndarray
            1D array containing the weight of each particle.

        Raises
        ------
        CoordinateError: Exception
            X-coordinates is None
        WaterfallReducedToZero: Exception
            All of reconstructed waterfall reduced to zero.
        """
        log.warning('TomographyCpp.run_hybrid() '
                    'may be removed in future updates!')
        if self.xp is None:
            raise expt.CoordinateError(
                'x-coordinates has value None, and must be provided')

        self.diff = np.zeros(niter + 1)
        reciprocal_pts = self._reciprocal_particles()
        flat_points = self._create_flat_points()
        flat_profs = np.ascontiguousarray(
            self.waterfall.flatten()).astype(np.float64)
        weight = np.zeros(self.nparts)

        weight = libtomo.back_project(weight, flat_points, flat_profs,
                                      self.nparts, self.nprofs)
        weight = weight.clip(0.0)

        if verbose:
            print(' Iterating...')

        for i in range(niter):
            if verbose:
                print(f'{i + 1:3d}')

            self.recreated = self._project(flat_points, weight)

            diff_waterfall = self.waterfall - self.recreated
            self.diff[i] = self._discrepancy(diff_waterfall)

            # Weighting difference waterfall relative to number of particles
            diff_waterfall *= reciprocal_pts.T

            weight = libtomo.back_project(
                weight, flat_points, diff_waterfall.flatten(),
                self.nparts, self.nprofs)
            weight = weight.clip(0.0)

        self.recreated = self._project(flat_points, weight)

        # Calculating final discrepancy
        diff_waterfall = self.waterfall - self.recreated
        self.diff[-1] = self._discrepancy(diff_waterfall)

        if verbose:
            print(' Done!')

        return weight

    # Project using C++ routine from tomolib_wrappers.
    # Normalizes recreated profiles before returning them.
    def _project(self, flat_points: np.ndarray, weight: np.ndarray) \
            -> np.ndarray:
        # rec = tlw.project(np.zeros(self.recreated.shape), flat_points,
        #                   weight, self.nparts, self.nprofs, self.nbins)
        rec = libtomo.project(np.zeros(self.recreated.shape), flat_points,
                              weight, self.nparts, self.nprofs, self.nbins)
        rec = self._normalize_profiles(rec)
        return rec

    # Convert x coordinates pointing at bins of flattened version of waterfall.
    def _create_flat_points(self) -> np.ndarray:
        return np.ascontiguousarray(
            super()._create_flat_points()).astype(np.int32)

    def _run_old(self, niter: int = 20, verbose: bool = False) -> np.ndarray:
        """Function to perform tomographic reconstruction.

        Performs the full reconstruction using C++.

        The discrepancy of each iteration is saved in the *diff* variable
        of the object.

        Parameters
        ----------
        niter: int
            Number of iterations in reconstruction.
        verbose: boolean
            Flag to indicate that the status of the tomography should be
            written to stdout. The output is identical to output
            generated in the original Fortran tomography.

        Returns
        -------
        weight: ndarray
            1D array containing the weight of each particle.

        Raises
        ------
        CoordinateError: Exception
            X-coordinates is None
        """
        if self.xp is None:
            raise expt.CoordinateError(
                'x-coordinates has value None, and must be provided')

        weight = np.ascontiguousarray(
            np.zeros(self.nparts, dtype=np.float64))
        self.diff = np.zeros(niter + 1, dtype=np.float64)

        flat_profiles = np.ascontiguousarray(
            self.waterfall.flatten().astype(np.float64))

        (self.weight,
         self.diff) = libtomo.reconstruct_old(
            weight, self.xp, flat_profiles,
            self.diff, niter, self.nbins,
            self.nparts, self.nprofs, verbose)
        return self.weight

    def run(self, niter: int = 20, verbose: bool = False,
            callback: t.Callable = None) -> np.ndarray:
        """Function to perform tomographic reconstruction.

        Performs the full reconstruction using C++.

        - The discrepancy of each iteration is saved in the objects\
          **diff** variable.
        - The particles weights are saved in the objects **weight** variable.
        - The reconstructed profiles are saved to the objects\
          **recreated** variable.

        Parameters
        ----------
        niter: int
            Number of iterations in reconstruction.
        verbose: boolean
            Flag to indicate that the status of the tomography should be
            written to stdout. The output is identical to output
            generated in the original Fortran tomography.
        callback: Callable
            Passing a callback with function signature
            (progress: int, total: int) will allow the tracking loop to call
            this function at the end of each turn, allowing the python caller
            to monitor the progress.

        Returns
        -------
        weight: ndarray
            1D array containing the weight of each particle.

        Raises
        ------
        CoordinateError: Exception
            X-coordinates is None
        """
        if self.xp is None:
            raise expt.CoordinateError(
                'x-coordinates has value None, and must be provided')

        (self.weight,
         self.diff,
         self.recreated) = libtomo.reconstruct(
            self.xp, self.waterfall, niter, self.nbins,
            self.nparts, self.nprofs, verbose, callback)
        return self.weight


# for backwards compatibility
TomographyCpp = Tomography
