import numpy as np
import ctypes
from cpp_routines.tomolib_wrappers import back_project, project


class NewTomographyC:

    def __init__(self, timespace, tracked_xp, tracked_yp):
        self.ts = timespace
        self.tracked_xp = tracked_xp - 1  # Fortran compensation
        self.tracked_yp = tracked_yp
        self.recreated = np.zeros((self.ts.par.profile_count,
                                   self.ts.par.profile_length))
        self.diff = np.zeros(self.ts.par.num_iter + 1)

    def run_cpp(self):
        nparts = self.tracked_xp.shape[0]
        weights = np.zeros(nparts)

        flat_points = self._create_flat_points()

        flat_profs = np.ascontiguousarray(self.ts.profiles.flatten()
                                          ).astype(ctypes.c_double)

        weights = back_project(weights, flat_points, flat_profs, nparts,
                               self.ts.par.profile_count)

        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            self.recreated = project(self.recreated, flat_points, weights,
                                     nparts, self.ts.par.profile_count,
                                     self.ts.par.profile_length)

            self.recreated /= np.sum(self.recreated, axis=1)[:, None]

            diff_prof = self.ts.profiles - self.recreated
            self.diff[i] = self.discrepancy(diff_prof)

            weights = back_project(weights, flat_points, diff_prof, nparts,
                                   self.ts.par.profile_count)

            weights = weights.clip(0.0)

        # Calculating final discrepancy
        diff_prof = self.ts.profiles - self.recreated
        self.diff[-1] = self.discrepancy(diff_prof)

        return weights

    def discrepancy(self, diff_profiles):
        return np.sqrt(np.sum(diff_profiles**2)/(self.ts.par.profile_length
                                                 * self.ts.par.profile_count))

    def _create_flat_points(self):
        flat_points = self.tracked_xp.copy()
        for i in range(self.ts.par.profile_count):
            flat_points[:, i] += self.ts.par.profile_length * i
        return np.ascontiguousarray(flat_points).astype(ctypes.c_int)
