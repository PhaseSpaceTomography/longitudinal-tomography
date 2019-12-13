import logging as log
from numba import njit
import numpy as np

from ..utils import exceptions as expt


class Tomography:

    # Needs measured proflies as two-dimensional array (waterfall).
    # X-coordinates should be integers, pointing bins in the x axis.
    # X-coordinates cannot be longer than the number of bins in the
    # waterfall. This will raise an exception.
    # 
    # Variables:
    # nparts    - number of test partices
    # nprofs    - number of profile measurements
    # nbins     - number of bins in profile measurement (image length).
    # waterfall - measured profiles as a 2D array. shape: (nprofs, nbins)
    # xp        - x coordinates as integers, pointig at bins of waterfall.
    #             shape: (nparts, nprof)
    # recreated - Recreated waterfall from phase-space back-projections
    # diff      - Array containing discrepancy for each iteration of
    #             recontruction.
    def __init__(self, waterfall, x_coords=None):
        self.waterfall = waterfall
        self.waterfall = self._suppress_zeros_normalize(self.waterfall)
        
        self._nprofs = self.waterfall.shape[0]
        self._nbins = self.waterfall.shape[1]

        self.xp = x_coords

        self.recreated = np.zeros(self.waterfall.shape)
        self.diff = None

    @property
    def xp(self):
        return self._xp

    @xp.setter    
    def xp(self, value):
        if hasattr(value, '__iter__'):
            value = np.array(value)
            if not value.ndim == 2:
                msg = 'X coordinates have two dimensions '\
                      '(nparticles, nprofiles)'
                raise expt.CoordinateImportError(msg)
            if not value.shape[1] == self.nprofs:
                msg = f'Imported particles should be '\
                      f'tracked trough {self.nprofs} profiles. '\
                      f'Given particles seems so have been tracked trough '\
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
    def nparts(self):
        return self._nparts

    @property
    def nbins(self):
        return self._nbins
    
    @property
    def nprofs(self):
        return self._nprofs
    
    def _suppress_zeros_normalize(self, waterfall):
        waterfall = waterfall.clip(0.0)
        if not waterfall.any():
            raise expt.WaterfallReducedToZero()
        waterfall /= np.sum(waterfall, axis=1)[:, None]
        return waterfall

    # Calculates discrepancy for the whole waterfall
    def _discrepancy(self, diff_waterfall):
        return np.sqrt(np.sum(diff_waterfall**2)/(self.nbins * self.nprofs))

    # xp array modified to point at flattened version of waterfall.
    def _create_flat_points(self):
        flat_points = self.xp.copy()
        for i in range(self.nprofs):
            flat_points[:, i] += self.nbins * i
        return flat_points
    
    # Finding the reciprocal of the number of particles in
    # a bin. Done to counterbalance the different amount of
    # particles in the different bins.
    # Bins with fewer particles have a larger amplification,
    # relative to bins containing many particles.
    def _reciprocal_particles(self):
        ppb = np.zeros((self.nbins, self.nprofs))
        ppb = self.count_particles_in_bins(
                ppb, self.nprofs, self.xp, self.nparts)
        
        # Setting zeros to one to avoid division by zero
        ppb[ppb==0] = 1
        return np.max(ppb) / ppb

    @staticmethod
    @njit
    # Static needed for use of njit.
    # To be replaced by C++ function?
    def count_particles_in_bins(ppb, profile_count, xp, nparts):
        for i in range(profile_count):
            for j in range(nparts):
                ppb[xp[j, i], i] += 1
        return ppb
