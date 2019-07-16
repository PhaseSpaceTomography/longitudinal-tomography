import matplotlib.pyplot as plt
import numpy as np
import time as tm
import logging as lg
from numba import njit, prange


class NewTomography:

    def __init__(self, timespace, tracked_points):
        self.ts = timespace
        self.tracked_points = tracked_points - 1  # Fortran compensation
        self.weights = np.zeros((self.ts.par.profile_count,
                                 tracked_points.shape[0]))
        self.bins = np.zeros((self.ts.par.profile_length,
                              self.ts.par.profile_count))

    def run(self):

        diff_prof = np.zeros(self.ts.profiles.shape)

        self.back_project_njit(self.tracked_points,
                               self.ts.profiles,
                               self.weights,
                               self.ts.par.profile_count)

        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            t0 = tm.perf_counter()
            self.project_njit(self.tracked_points, self.bins,
                              self.weights,
                              self.ts.par.profile_count)
            lg.info('project_t: ' + str(tm.perf_counter() - t0))

            self.weights[:, :] = 0

            t0 = tm.perf_counter()
            for p in range(self.ts.par.profile_count):
                diff_prof[p] = ((self.ts.profiles[p]
                                 / np.sum(self.ts.profiles[p]))
                                - (self.bins[:, p]
                                / np.sum(self.bins[:, p])))
            lg.info('difference_t: ' + str(tm.perf_counter() - t0))

            lg.info(f'discrepancy: {self.discrepancy(diff_prof)}')

            t0 = tm.perf_counter()
            self.back_project_njit(self.tracked_points,
                                   diff_prof,
                                   self.weights,
                                   self.ts.par.profile_count)
            lg.info('back_projection_t: ' + str(tm.perf_counter() - t0))

        # TEMP
        # self.analyze(profilei=0, diff_prof=diff_prof)
        # END TEMP

    @staticmethod
    @njit(parallel=True)
    def project_njit(tracked_points, bins, weights, profile_count):
        for profile in prange(profile_count):
            for point in range(tracked_points.shape[0]):
                bins[tracked_points[point, profile],
                     profile] += weights[profile, point]

    @staticmethod
    @njit(parallel=True)
    def back_project_njit(tracked_points,
                           profiles,
                           weight_factors,
                           profile_count):

        for p in prange(profile_count):
            counter = 0
            for point in tracked_points[:, p]:
                weight_factors[p, counter] += profiles[p, point]
                counter += 1

    def discrepancy(self, diff_profiles):
        return np.sqrt(np.sum(diff_profiles**2)/(self.ts.par.profile_length
                                                 * self.ts.par.profile_count))

    # TEMP
    def analyze(self, profilei, diff_prof):
        plt.figure()

        # Plotting profiles
        plt.subplot(311)
        plt.title('Profiles')
        plt.plot(self.ts.profiles[profilei] / np.sum(self.ts.profiles[profilei]))
        plt.plot(self.bins[:, profilei] / np.sum(self.bins[:, profilei]))
        plt.gca().legend(('original', 'recreated python'))

        # Plotting difference
        plt.subplot(313)
        plt.title('Difference')
        plt.plot(diff_prof[profilei])
        plt.show()
    # END TEMP
