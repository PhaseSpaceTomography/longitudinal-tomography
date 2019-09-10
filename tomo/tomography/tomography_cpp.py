import numpy as np
import ctypes
from cpp_routines import tomolib_wrappers as tlw
from tomography.__tomography import Tomography

class TomographyCpp(Tomography):

    def __init__(self, timespace, tracked_xp, tracked_yp):
        super().__init__(timespace, tracked_xp, tracked_yp)

    def run(self):
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

    # Wrapper for projecting using cpp (calling fraom tomolib_wrappers module)
    def project(self, flat_points, weight, nparts):
        rec = tlw.project(np.zeros(self.recreated.shape), flat_points, weight,
                          nparts, self.ts.par.profile_count,
                          self.ts.par.profile_length)
        
        rec = self._suppress_zeros_normalize(rec)
        return rec

    def _create_flat_points(self):
        return np.ascontiguousarray(
                super()._create_flat_points()).astype(ctypes.c_int)

