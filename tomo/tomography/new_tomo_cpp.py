import numpy as np
from numba import njit
import ctypes
from cpp_routines import tomolib_wrappers as tlw

class NewTomographyC:

    def __init__(self, timespace, tracked_xp, tracked_yp):
        self.ts = timespace
        self.xp = tracked_xp
        self.yp = tracked_yp
        self.recreated = np.zeros((self.ts.par.profile_count,
                                   self.ts.par.profile_length))
        self.diff = np.zeros(self.ts.par.num_iter + 1)

    def run_cpp(self):
        nparts = self.xp.shape[0]

        weight = np.zeros(nparts)

        reciprocal_pts = self.reciprocal_particles(nparts)

        flat_points = self._create_flat_points()

        flat_profs = np.ascontiguousarray(self.ts.profiles.flatten()
                                          ).astype(ctypes.c_double)

        weight = tlw.back_project(weight, flat_points, flat_profs, nparts,
                                 self.ts.par.profile_count)

        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            self.recreated = self.project(flat_points, weight, nparts)

            diff_prof = self.ts.profiles - self.recreated
            self.diff[i] = self.discrepancy(diff_prof)

            # Weighting difference profiles relative to number of particles
            diff_prof *= reciprocal_pts.T
         
            weight = tlw.back_project(weight, flat_points, diff_prof, nparts,
                                      self.ts.par.profile_count)

        self.recreated = self.project(flat_points, weight, nparts)

        # Calculating final discrepancy
        diff_prof = self.ts.profiles - self.recreated
        self.diff[-1] = self.discrepancy(diff_prof)

        return weight

    def project(self, flat_points, weight, nparts):
        rec = tlw.project(np.zeros(self.recreated.shape), flat_points, weight,
                          nparts, self.ts.par.profile_count,
                          self.ts.par.profile_length)
        
        # Normalizing and removing zeros
        rec = rec.clip(0.0)
        rec /= np.sum(rec, axis=1)[:, None]
        return rec

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
        return np.ascontiguousarray(flat_points).astype(ctypes.c_int)

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
