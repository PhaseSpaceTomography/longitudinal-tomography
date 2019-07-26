import numpy as np
import ctypes


class NewTomographyC:

    def __init__(self, timespace, tracked_xp, tracked_yp):
        self.ts = timespace
        self.tracked_xp = tracked_xp - 1  # Fortran compensation
        self.tracked_yp = tracked_yp
        self.recreated = np.zeros((self.ts.par.profile_count,
                                   self.ts.par.profile_length))
        self.diff = np.zeros(self.ts.par.num_iter + 1)

        # Setting up C++ functions
        double_ptr = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')
        # tomolib_path = './tomo/cpp_files/tomolib.so'
        tomolib_path = './cpp_files/tomolib.so'
        tomolib = ctypes.CDLL(tomolib_path)

        self._back_proj = tomolib.back_project
        self._back_proj.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double),
                                    double_ptr,
                                    np.ctypeslib.ndpointer(ctypes.c_double)]
        self._back_proj.restypes = None

        self._proj = tomolib.project
        self._proj.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double),
                               double_ptr,
                               np.ctypeslib.ndpointer(ctypes.c_double)]
        self._proj.restypes = None

    def run_cpp(self):
        nparts = self.tracked_xp.shape[0]
        weights = np.zeros(nparts)
        flat_points = self.tracked_xp.copy()
        for i in range(self.ts.par.profile_count):
            flat_points[:, i] += self.ts.par.profile_length * i

        flat_profs = np.ascontiguousarray(
                        self.ts.profiles.flatten()).astype(ctypes.c_double)
        flat_points = np.ascontiguousarray(flat_points).astype(ctypes.c_int)

        self._back_proj(weights, self._get_2d_pointer(flat_points),
                        flat_profs, nparts, self.ts.par.profile_count)

        # diff_prof = None
        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            self.recreated = np.ascontiguousarray(self.recreated.flatten())
            self._proj(self.recreated,
                       self._get_2d_pointer(flat_points),
                       weights, nparts, self.ts.par.profile_count)

            self.recreated = self.recreated.reshape(self.ts.profiles.shape)

            self.recreated /= np.sum(self.recreated, axis=1)[:, None]

            diff_prof = self.ts.profiles - self.recreated
            self.diff[i] = self.discrepancy(diff_prof)

            self._back_proj(weights, self._get_2d_pointer(flat_points),
                            diff_prof, nparts, self.ts.par.profile_count)

            weights = np.where(weights < 0.0, 0.0, weights)

        # Calculating final discrepancy
        diff_prof = self.ts.profiles - self.recreated
        self.diff[-1] = self.discrepancy(diff_prof)

        return weights

    def discrepancy(self, diff_profiles):
        return np.sqrt(np.sum(diff_profiles**2)/(self.ts.par.profile_length
                                                 * self.ts.par.profile_count))

    @ staticmethod
    def _get_2d_pointer(arr2d):
        return (arr2d.__array_interface__['data'][0]
                + np.arange(arr2d.shape[0]) * arr2d.strides[0]).astype(np.uintp)
