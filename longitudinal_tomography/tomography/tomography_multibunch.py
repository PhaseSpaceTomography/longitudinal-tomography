"""Module containing the multibunch tomography class

:Author(s): **Simon Albright**
"""

import logging
import time as tm
import typing as t
import sys

import numpy as np
import matplotlib.pyplot as plt

from .__tomography import TomographyABC
from ..cpp_routines import libtomo
from .. import exceptions as expt

log = logging.getLogger(__name__)


class Tomography(TomographyABC):
    """Class for performing tomographic reconstruction of multibunch phase
    space.

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


    def run_hybrid(self, centers, cuts, niter=20, verbose=False):
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

        nBunches = len(centers)

        masks = []
        for cent, cut in zip(centers, cuts):
            mask = np.all(((self.xp+cent)>cut[0])*((self.xp+cent)<cut[1]),
                          axis=1)
            masks += mask.tolist()
        masks = np.array(masks)

        self.diff = np.zeros(niter + 1)
        self.diff_split = np.zeros([nBunches, niter + 1])

        reciprocal_pts = self._reciprocal_particles_multi(centers)
        flat_points = self._create_flat_points()

        flat_profs = np.ascontiguousarray(
            self.waterfall.flatten()).astype(np.float64)
        weight = np.zeros(self.nparts*nBunches)

        for i in range(nBunches):
            start = i*self.nparts
            stop = (i+1)*self.nparts
            weight[start:stop][masks[start:stop]] = libtomo.back_project(
                                                weight[start:stop][masks[start:stop]],
                                                flat_points[masks[start:stop]] \
                                                    + centers[i],
                                                flat_profs,
                                                np.sum(masks[start:stop]),
                                                self.nprofs)
        weight = weight.clip(0.0)

        if verbose:
            print(' Iterating...')

        for i in range(niter):
            if verbose:
                print(f'{i + 1:3d}')

            self.full_recreated = np.zeros_like(self.waterfall)

            for j in range(nBunches):
                start = j*self.nparts
                stop = (j+1)*self.nparts
                self.recreated = self._project(flat_points + centers[j],
                                               weight[start:stop],
                                               self.nparts)

                self.full_recreated += self.recreated

            self.recreated = self._normalize_profiles(self.full_recreated)

            diff_waterfall = self.waterfall - self.recreated
            self.diff[i] = self._discrepancy(diff_waterfall)
            self.diff_split[:, i] = self._discrepancy_multi(diff_waterfall, cuts)

            # Weighting difference waterfall relative to number of particles
            diff_waterfall *= reciprocal_pts

            for j in range(nBunches):
                start = j*self.nparts
                stop = (j+1)*self.nparts
                
                weight[start:stop][masks[start:stop]] = libtomo.back_project(
                                            weight[start:stop][masks[start:stop]],
                                            flat_points[masks[start:stop]] + centers[j],
                                            diff_waterfall.flatten(),
                                            np.sum(masks[start:stop]), self.nprofs)

            weight = weight.clip(0.0)

        self.full_recreated = np.zeros_like(self.waterfall)
        
        for j in range(nBunches):
            start = j*self.nparts
            stop = (j+1)*self.nparts
            self.recreated = self._project(flat_points + centers[j],
                                           weight[start:stop], self.nparts)

            self.full_recreated += self.recreated
        self.recreated = self._normalize_profiles(self.full_recreated)

        # Calculating final discrepancy
        diff_waterfall = self.waterfall - self.recreated
        self.diff[-1] = self._discrepancy(diff_waterfall)
        self.diff_split[:, -1] = self._discrepancy_multi(diff_waterfall, cuts)
        
        if verbose:
            print(' Done!')


        self.weight_combined = weight
        self.weight_split = []
        for i in range(nBunches):
            start = i*self.nparts
            stop = (i+1)*self.nparts
            self.weight_split.append(weight[start:stop])



    def _discrepancy_multi(self, diff_waterfall, cuts):

        allDiffs = []
        for c in cuts:
            cutWFall = diff_waterfall[:,c[0]:c[1]]
            diff = np.sqrt(np.sum(cutWFall**2)/((c[1]-c[0])*self.nprofs))
            allDiffs.append(diff)

        return allDiffs


    def _project(self, flat_points: np.ndarray, weight: np.ndarray,
                       nUseParts: int) -> np.ndarray:

        rec = libtomo.project(np.zeros(self.recreated.shape), flat_points,
                              weight, nUseParts, self.nprofs, self.nbins)

        return rec

    # Convert x coordinates pointing at bins of flattened version of waterfall.
    def _create_flat_points(self) -> np.ndarray:
        return np.ascontiguousarray(
            super()._create_flat_points()).astype(np.int32)


    def run(self, centers: [int], cutleft: [int], cutright: [int],
            niter: int = 20, verbose: bool = False,
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

        # (self.weight,
        #  self.diff,
        #  self.recreated) = libtomo.reconstruct(
        #     self.xp, self.waterfall, niter, self.nbins,
        #     self.nparts, self.nprofs, verbose, callback)
        
        (weight, self.diff, self.diff_split, self.recreated) = \
            libtomo.reconstruct_multi(self.xp, self.waterfall, cutleft,
                                      cutright, centers, niter, self.nbins,
                                      self.nparts, self.nprofs, len(centers),
                                      verbose, callback)
        
        self.diff_split = self.diff_split.reshape([niter+1, len(centers)]).T
        
        self.weight_combined = weight
        self.weight_split = []
        for i in range(len(centers)):
            start = i*self.nparts
            stop = (i+1)*self.nparts
            self.weight_split.append(weight[start:stop])

