import numpy as np
from numba import njit, prange

# This class is using python only, speed up with numba. 

class NewTomography:

    def __init__(self, timespace, tracked_xp, tracked_yp):
        self.ts = timespace
        self.xp = tracked_xp
        self.yp = tracked_yp
        self.recreated = np.zeros((self.ts.par.profile_count,
                                   self.ts.par.profile_length))
        self.diff = np.zeros(self.ts.par.num_iter + 1)

    def run(self):
        nparts = self.xp.shape[0]
        weights = np.zeros(nparts)
        flat_points = self.xp.copy()

        pt_wgts = self.parts_pr_bin(nparts)

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

            diff_prof *= pt_wgts.T

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
        rec = rec.clip(0.0)
        rec /= np.sum(rec, axis=1)[:, None]
        return rec

    # Counting number of particles per bin
    def parts_pr_bin(self, nparts):
        ppb = np.zeros((self.ts.par.profile_length,
                        self.ts.par.profile_count))
        ppb = self.count_bins(
                  ppb, self.ts.par.profile_count,
                  self.xp, nparts)
        
        ppb[ppb==0] = 1

        return np.max(ppb) / ppb

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

    @staticmethod
    @njit
    def count_bins(ppb, profile_count, xp, nparts):
        for i in range(profile_count):
            for j in range(nparts):
                ppb[xp[j, i], i] += 1
        return ppb

    def discrepancy(self, diff_profiles):
        return np.sqrt(np.sum(diff_profiles**2)/(self.ts.par.profile_length
                                                 * self.ts.par.profile_count))
