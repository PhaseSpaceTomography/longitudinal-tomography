'''Module containing the Tomography class

:Author(s): **Christoffer Hjertø Grindheim**
'''

import logging as log
from numba import njit
import numpy as np

from ..utils import exceptions as expt


class Tomography:
    '''Base class for tomography classes.

    This class holds tomography utlities and assertions.

    Parameters
    ----------
    waterfall: ndarray, float
        Measured profiles (nprofiles, nbins).
        Negative values of waterfall is set to zero, and the
        waterfall is normalized as the tomography objects are created.   
    x_coords: ndarray: int
        particles x-coordinates, given as coordinates of the reconstructed
        phase space coordinate system (nparts, nprofiles).

    Attributes
    ----------
    nparts: int
        Number of test particles.
    nprofs: int
        Number of profiles (time frames).
    nbins: int
        Number of bins in each profile.
    waterfall: ndarray, float
        Measured profiles.
    xp: ndarray, int
        particles x-coordinates, given as coordinates of the reconstructed
        phase space coordinate system (nparts, nprofiles).
    recreated: ndarray, float
        Recreated waterfall. Directly comparable with `waterfall`.
    diff: ndarray, float
        Discrepancy for phase space reconstruction at each iteration
        of the reconstruction process.
    '''
    def __init__(self, waterfall, x_coords=None):
        self._waterfall = waterfall.clip(0.0)
        self._waterfall = self._normalize_profiles(self.waterfall)
        
        self._nprofs = self.waterfall.shape[0]
        self._nbins = self.waterfall.shape[1]

        self.xp = x_coords

        self.recreated = np.zeros(self.waterfall.shape)
        self.diff = None

    @property
    def waterfall(self):
        '''Waterfall defined as @property.

        Returns
        -------
        waterfall: ndarray, float
            Measured profiles (nprofiles, nbins).
            Negative values of waterfall is set to zero, and the
            waterfall is normalized as the tomography objects are created. 
        '''
        return self._waterfall
    
    @property
    def xp(self):
        '''X-coordinates defined as @property.

        Automaticly updates `nparts`.

        Parameters
        ----------
        value: ndarray, int, None
            Tracked particles, given in phase space coordinates as integers.
            By setting tomography.xp = None,
            the saved x-coordinates are deleted.

        Returns
        -------
        xp: ndarray, int
            particles x-coordinates, given as coordinates of the reconstructed
            phase space coordinate system (nparts, nprofiles).

        Raises
        ------
        CoordinateImportError: Exception
            Provided coordinates are invalid.
        XPOutOfImageWidthError: Exception
            Particle(s) have left the image width.
        '''
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
        '''Number of particles defined as @property.
        
        Returns
        -------
        nparts: int
            Number of test particles.
        '''
        return self._nparts

    @property
    def nbins(self):
        '''Number of profile bins defined as @property.
        
        Returns
        -------
        nbins: int
            Number of bins in each profile.
        '''
        return self._nbins
    
    @property
    def nprofs(self):
        '''Number of profiles defined as @property.

        nprofs: int
            Number of profiles (time frames).
        '''
        return self._nprofs
    
    def _normalize_profiles(self, waterfall):
        if not waterfall.any():
            raise expt.WaterfallReducedToZero()
        waterfall /= np.sum(waterfall, axis=1)[:, None]
        return waterfall

    # Calculates discrepancy for the whole waterfall
    def _discrepancy(self, diff_waterfall):
        return np.sqrt(np.sum(diff_waterfall**2)/(self.nbins * self.nprofs))

    # Created xp array modified to point at flattened version of waterfall.
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
        
        # Setting bins with zero particles one to avoid division by zero.
        ppb[ppb==0] = 1
        return np.max(ppb) / ppb

    @staticmethod
    @njit
    # Static needed for use of njit.
    # Needed by reciprocal paricles function.
    # C++ version excists.
    def count_particles_in_bins(ppb, profile_count, xp, nparts):
        for i in range(profile_count):
            for j in range(nparts):
                ppb[xp[j, i], i] += 1
        return ppb
