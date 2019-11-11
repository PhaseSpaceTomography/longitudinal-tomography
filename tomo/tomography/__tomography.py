import numpy as np
from numba import njit
from utils.exceptions import (PhaseSpaceReducedToZeroes,
                              XPOutOfImageWidthError)

class Tomography:

    def __init__(self, profiles, x_coords):
        self.profiles = profiles
        self.profiles = self._suppress_zeros_normalize(self.profiles)
        
        self.nprofs = self.profiles.shape[0]
        self.nbins = self.profiles.shape[1]

        self.xp = x_coords
        self.assert_xp()

        self.nparts = self.xp.shape[0]

        self.recreated = np.zeros(self.profiles.shape)
        self.diff = None


    def _suppress_zeros_normalize(self, profiles):
        profiles = profiles.clip(0.0)
        if not profiles.any():
            raise PhaseSpaceReducedToZeroes(
                    'All of phase space got reduced to zeroes')
        profiles /= np.sum(profiles, axis=1)[:, None]
        return profiles

    def _discrepancy(self, diff_profiles):
        return np.sqrt(np.sum(diff_profiles**2)/(self.nbins * self.nprofs))

    # To be clear: The array is of the xp's are not flat, but
    # the xp values of the 'flat xp' points at the correct bin
    # in the actual flattened one dimensional profile array 
    def _create_flat_points(self):
        flat_points = self.xp.copy()
        for i in range(self.nprofs):
            flat_points[:, i] += self.nbins * i
        return flat_points
    
    # Finding the reciprocal of the number of particles in
    # a bin, to counterbalance the different amount of
    # particles in the different bins
    def _reciprocal_particles(self):
        ppb = np.zeros((self.nbins, self.nprofs))
        ppb = self.count_particles_in_bins(
                ppb, self.nprofs, self.xp, self.nparts)
        
        # Setting zeros to one to avoid division by zero
        ppb[ppb==0] = 1
        return np.max(ppb) / ppb

    def assert_xp(self):
        # Checking that no particles are outside of image width
        if np.any(self.xp < 0) or np.any(self.xp >= self.nbins):
            raise XPOutOfImageWidthError(
                'X coordinate of particles outside of image width')

    @staticmethod
    @njit
    def count_particles_in_bins(ppb, profile_count, xp, nparts):
        for i in range(profile_count):
            for j in range(nparts):
                ppb[xp[j, i], i] += 1
        return ppb
