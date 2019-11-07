import numpy as np
import ctypes
from cpp_routines import tomolib_wrappers as tlw
from tomography.__tomography import Tomography

class TomographyCpp(Tomography):

    def __init__(self, profiles, x_coords):
        super().__init__(profiles, x_coords)

    # Hybrid Python/C++ coutine.
    # Back project and project routines are written in C++
    #  and are reached via the tomolib_wrappers module.
    def run(self, niter=20):
        self.diff = np.zeros(niter + 1)

        weight = np.zeros(self.nparts)

        reciprocal_pts = self._reciprocal_particles()

        flat_points = self._create_flat_points()

        flat_profs = np.ascontiguousarray(
                        self.profiles.flatten()).astype(ctypes.c_double)

        weight = tlw.back_project(
                    weight, flat_points, flat_profs, self.nparts, self.nprofs)

        print(' Iterating...')
        for i in range(niter):
            print(f'{i + 1:3d}')

            self.recreated = self.project(flat_points, weight)

            diff_prof = self.profiles - self.recreated
            self.diff[i] = self._discrepancy(diff_prof)

            # Weighting difference profiles relative to number of particles
            diff_prof *= reciprocal_pts.T
         
            weight = tlw.back_project(weight, flat_points, diff_prof,
                                      self.nparts, self.nprofs)

        self.recreated = self.project(flat_points, weight)

        # Calculating final discrepancy
        diff_prof = self.profiles - self.recreated
        self.diff[-1] = self._discrepancy(diff_prof)

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
                super()._create_flat_points()).astype(ctypes.c_int)


    # Running the full tomography routine in c++.
    # Must be tested with more inputs.
    def run_cpp(self, niter=20):
        weight = np.ascontiguousarray(
                    np.zeros(self.nparts, dtype=ctypes.c_double))
        self.diff = np.zeros(niter + 1, dtype=ctypes.c_double)
        self.xp = np.ascontiguousarray(self.xp).astype(ctypes.c_int)

        flat_profs = np.ascontiguousarray(
                        self.profiles.flatten().astype(ctypes.c_double))

        weight = tlw.reconstruct(
                    weight, self.xp, flat_profs, self.diff, 
                    niter, self.nbins, self.nparts, self.nprofs)

        return weight
