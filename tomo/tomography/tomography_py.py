import numpy as np
from numba import njit, prange
from tomography.__tomography import Tomography

# This class is using python only, speed up with numba. 
class TomographyPy(Tomography):

    def __init__(self, timespace, tracked_xp, tracked_yp):
        super().__init__(timespace, tracked_xp, tracked_yp)

    def run(self):
        nparts = self.xp.shape[0]
        weights = np.zeros(nparts)
        flat_points = self.xp.copy()

        reciprocal_pts = self.reciprocal_particles(nparts)

        for i in range(self.ts.par.profile_count):
            flat_points[:, i] += self.ts.par.profile_length * i

        weights = self.back_project_flattened(self.ts.profiles.flatten(),
                                              flat_points,
                                              weights, nparts)

        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            self.recreated = self.project(flat_points, weights, nparts)

            diff_prof = self.ts.profiles - self.recreated
            self.diff[i] = self.discrepancy(diff_prof)

            diff_prof *= reciprocal_pts.T

            weights = self.back_project_flattened(diff_prof.flatten(),
                                                  flat_points,
                                                  weights, nparts)
        
        self.recreated = self.project(flat_points, weights, nparts)

        diff_prof = self.ts.profiles - self.recreated
        self.diff[-1] = self.discrepancy(diff_prof)
        
        return weights

    def project(self, flat_points, weights, nparts):
        rec = self._project_flattened(self.recreated.flatten(),
                                      flat_points, weights, nparts)
        rec = rec.reshape(self.ts.profiles.shape)
        rec = self._suppress_zeros_normalize(rec)
        return rec

    @staticmethod
    @njit(parallel=True)
    def back_project_flattened(flat_profiles, flat_points,
                                   weights, nparts):
        for i in prange(nparts):
            weights[i] += np.sum(flat_profiles[flat_points[i]])
        return weights

    @staticmethod
    @njit
    def _project_flattened(flat_rec, flat_points, weights, nparts):
        for i in range(nparts):
            flat_rec[flat_points[i]] += weights[i]
        return flat_rec
