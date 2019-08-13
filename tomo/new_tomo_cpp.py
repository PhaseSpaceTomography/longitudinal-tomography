import numpy as np
from numba import njit
import ctypes
from cpp_routines.tomolib_wrappers import back_project, project

class NewTomographyC:

    def __init__(self, timespace, tracked_xp, tracked_yp):
        self.ts = timespace
        self.tracked_xp = tracked_xp - 1  # Fortran compensation
        self.tracked_yp = tracked_yp - 1
        self.recreated = np.zeros((self.ts.par.profile_count,
                                   self.ts.par.profile_length))
        self.diff = np.zeros(self.ts.par.num_iter + 1)

    def run_cpp(self):
        nparts = self.tracked_xp.shape[0]
        weight = np.zeros(nparts)

        # Counting numer of particles per each bin.
        ppb = np.zeros((self.ts.par.profile_length,
                        self.ts.par.profile_count))
        ppb = self.count_bins(
                  ppb, self.ts.par.profile_count,
                  self.tracked_xp, nparts)
        ppb[ppb==0] = 1
        pt_wgts = np.max(ppb) / ppb

        flat_points = self._create_flat_points()

        flat_profs = np.ascontiguousarray(self.ts.profiles.flatten()
                                          ).astype(ctypes.c_double)

        weight = back_project(weight, flat_points, flat_profs, nparts,
                               self.ts.par.profile_count)

        phase_space = np.zeros((self.ts.par.profile_length,
                                self.ts.par.profile_length))
        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            self.recreated = project(self.recreated, flat_points, weight,
                                     nparts, self.ts.par.profile_count,
                                     self.ts.par.profile_length)

            # Normalizing and removing zeros
            self.recreated /= np.sum(self.recreated, axis=1)[:, None]
            self.recreated = self.recreated.clip(0.0)

            diff_prof = self.ts.profiles - self.recreated
            self.diff[i] = self.discrepancy(diff_prof)

            # Weighting difference profiles relative to number of particles
            diff_prof *= pt_wgts.T

            # Backprojecting
            weight = back_project(weight, flat_points, diff_prof, nparts,
                                   self.ts.par.profile_count)

        # Calculating final discrepancy
        diff_prof = self.ts.profiles - self.recreated

        self.recreated = project(self.recreated, flat_points, weight,
                                 nparts, self.ts.par.profile_count,
                                 self.ts.par.profile_length)

        # Normalizing and removing zeros
        self.recreated /= np.sum(self.recreated, axis=1)[:, None]
        self.recreated = self.recreated.clip(0.0)

        self.diff[-1] = self.discrepancy(diff_prof)

        return weight


    def discrepancy(self, diff_profiles):
        return np.sqrt(np.sum(diff_profiles**2)/(self.ts.par.profile_length
                                                 * self.ts.par.profile_count))

    def _create_flat_points(self):
        flat_points = self.tracked_xp.copy()
        for i in range(self.ts.par.profile_count):
            flat_points[:, i] += self.ts.par.profile_length * i
        return np.ascontiguousarray(flat_points).astype(ctypes.c_int)

    @staticmethod
    @njit
    def count_bins(ppb, profile_count, xp, nparts):
        for i in range(profile_count):
            for j in range(nparts):
                ppb[xp[j, i], i] += 1
        return ppb
