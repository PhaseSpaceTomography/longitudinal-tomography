"""Module containing the TomographyCpp class

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""

import logging
import typing as t

import numpy as np
import cupy as cp

from .__tomography_cupy import TomographyCuPyABC
from ..cpp_routines import libtomo
from .. import exceptions as expt
from ..utils.execution_mode import Mode
from longitudinal_tomography.python_routines.reconstruct_cupy import reconstruct_cupy
from longitudinal_tomography.python_routines.reconstruct_cuda import reconstruct_cuda

log = logging.getLogger(__name__)


class TomographyCuPy(TomographyCuPyABC):
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

    def __init__(self, waterfall: cp.ndarray, x_coords: cp.ndarray = None,
                 y_coords: cp.ndarray = None):
        super().__init__(waterfall, x_coords, y_coords)

    def run(self, niter: int = 20, verbose: bool = False,
            callback: t.Callable = None, mode: Mode = Mode.JIT) -> cp.ndarray:
        """Function to perform tomographic reconstruction.

        Performs the full reconstruction using CuPy.

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


        if mode == Mode.CUPY:
            (self.weight, self.diff, self.recreated) = reconstruct_cupy(
                self.xp, self.waterfall, niter, self.nbins,
                self.nparts, self.nprofs, verbose)
        else:
            (self.weight, self.diff, self.recreated) = reconstruct_cuda(
                self.xp, self.waterfall, niter, self.nbins,
                self.nparts, self.nprofs, verbose)

        return self.weight