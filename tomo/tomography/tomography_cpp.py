import numpy as np
import logging as log

from ..cpp_routines import tomolib_wrappers as tlw
from . import __tomography as stmo


class TomographyCpp(stmo.Tomography):

    def __init__(self, waterfall, x_coords=None):
        super().__init__(waterfall, x_coords)

    # Hybrid Python/C++ coutine.
    # Back project and project routines are written in C++
    #  and are reached via the tomolib_wrappers module.
    def run_hybrid(self, niter=20):
        log.warning('TomographyCpp.run_hybrid() '
                    'may be removed in future updates!')
        if self.xp is None:
            raise CoordinateError('No found x-coordinates.')
        
        self.diff = np.zeros(niter + 1)
        reciprocal_pts = self._reciprocal_particles()
        flat_points = self._create_flat_points()
        flat_profs = np.ascontiguousarray(
                        self.waterfall.flatten()).astype(np.float64)
        weight = np.zeros(self.nparts)
        weight = tlw.back_project(weight, flat_points, flat_profs,
                              self.nparts, self.nprofs)

        print(' Iterating...')
        for i in range(niter):
            print(f'{i + 1:3d}')

            self.recreated = self.project(flat_points, weight)

            diff_waterfall = self.waterfall - self.recreated
            self.diff[i] = self._discrepancy(diff_waterfall)

            # Weighting difference waterfall relative to number of particles
            diff_waterfall *= reciprocal_pts.T
         
            weight = tlw.back_project(
                        weight, flat_points, diff_waterfall.flatten(),
                        self.nparts, self.nprofs)

        self.recreated = self.project(flat_points, weight)

        # Calculating final discrepancy
        diff_waterfall = self.waterfall - self.recreated
        self.diff[-1] = self._discrepancy(diff_waterfall)

        print(' Done!')

        return weight

    # Project using c++ routine from tomolib_wrappers.
    # Normalizes and supresses zeroes.
    def project(self, flat_points, weight):
        rec = tlw.project(np.zeros(self.recreated.shape), flat_points,
                          weight, self.nparts, self.nprofs, self.nbins)
        
        rec = self._suppress_zeros_normalize(rec)
        return rec

    def _create_flat_points(self):
        return np.ascontiguousarray(
                super()._create_flat_points()).astype(np.int32)


    # Running the full tomography routine in c++.
    # Not as mature as run_hybrid()
    def run(self, niter=20):
        if self.xp is None:
            raise CoordinateError('No found x-coordinates.')
        
        weight = np.ascontiguousarray(
                    np.zeros(self.nparts, dtype=np.float64))
        self.diff = np.zeros(niter + 1, dtype=np.float64)

        diff_waterfall = np.ascontiguousarray(
                        self.waterfall.flatten().astype(np.float64))

        weight, self.diff = tlw.reconstruct(weight, self.xp, diff_waterfall,
                                            self.diff, niter, self.nbins,
                                            self.nparts, self.nprofs)
        return weight
