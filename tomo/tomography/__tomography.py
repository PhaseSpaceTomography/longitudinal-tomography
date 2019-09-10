import numpy as np
from numba import njit

class Tomography:

    def __init__(self, timespace, tracked_xp, tracked_yp):
        self.ts = timespace
        self.xp = tracked_xp
        self.yp = tracked_yp
        self.recreated = np.zeros((self.ts.par.profile_count,
                                   self.ts.par.profile_length))
        self.diff = np.zeros(self.ts.par.num_iter + 1)

    def discrepancy(self, diff_profiles):
        return np.sqrt(np.sum(diff_profiles**2)/(self.ts.par.profile_length
                                                 * self.ts.par.profile_count))

    # To be clear: The array is of the xp's are not flat, but
    # the xp values of the 'flat xp' points at the correct bin
    # in the actual flattened one dimensional profile array 
    def _create_flat_points(self):
        flat_points = self.xp.copy()
        for i in range(self.ts.par.profile_count):
            flat_points[:, i] += self.ts.par.profile_length * i
        return flat_points
    
    # Finding the reciprocal of the number of particles in
    # a bin, to counterbalance the different amount of
    # particles in the different bins
    def reciprocal_particles(self, nparts):
        ppb = np.zeros((self.ts.par.profile_length,
                        self.ts.par.profile_count))
        ppb = self.count_particles_in_bins(
                  ppb, self.ts.par.profile_count,
                  self.xp, nparts)
        
        # Setting zeros to one to avoid division by zero
        ppb[ppb==0] = 1
        return np.max(ppb) / ppb

    @staticmethod
    @njit
    def count_particles_in_bins(ppb, profile_count, xp, nparts):
        for i in range(profile_count):
            for j in range(nparts):
                ppb[xp[j, i], i] += 1
        return ppb
