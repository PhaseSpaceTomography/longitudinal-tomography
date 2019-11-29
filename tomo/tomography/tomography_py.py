import numpy as np
from numba import njit, prange

from . import __tomography as stmo

# This class is using python only, speed up with numba. 
class TomographyPy(stmo.Tomography):

    def __init__(self, waterfall, x_coords):
        super().__init__(waterfall, x_coords)

    def run(self, niter=20):
        self.diff = np.zeros(niter + 1)
        weights = np.zeros(self.nparts)
        flat_points = self.xp.copy()
        reciprocal_pts = self._reciprocal_particles()

        for i in range(self.nprofs):
            flat_points[:, i] += self.nbins * i

        weights = self.back_project_flattened(
                    self.waterfall.flatten(), flat_points,
                    weights, self.nparts)

        for i in range(niter):
            print(f'iteration: {str(i + 1)} of {niter}')

            self.recreated = self.project(flat_points, weights)

            diff_waterfall = self.waterfall - self.recreated
            self.diff[i] = self._discrepancy(diff_waterfall)

            diff_waterfall *= reciprocal_pts.T

            weights = self.back_project_flattened(
                        diff_waterfall.flatten(), flat_points,
                        weights, self.nparts)
        
        self.recreated = self.project(flat_points, weights)

        diff_waterfall = self.waterfall - self.recreated
        self.diff[-1] = self._discrepancy(diff_waterfall)
        
        return weights

    def project(self, flat_points, weights):
        rec = self._project_flattened(self.recreated.flatten(),
                                      flat_points, weights, self.nparts)
        rec = rec.reshape(self.waterfall.shape)
        rec = self._suppress_zeros_normalize(rec)
        return rec

    @staticmethod
    @njit(parallel=True)
    def back_project_flattened(flat_waterfall, flat_points,
                                   weights, nparts):
        for i in prange(nparts):
            weights[i] += np.sum(flat_waterfall[flat_points[i]])
        return weights

    @staticmethod
    @njit
    def _project_flattened(flat_rec, flat_points, weights, nparts):
        for i in range(nparts):
            flat_rec[flat_points[i]] += weights[i]
        return flat_rec
