import matplotlib.pyplot as plt
import numpy as np
import time as tm
import logging as lg
from numba import njit, prange


class NewTomography:

    def __init__(self, timespace, tracked_xp, tracked_yp):
        self.ts = timespace
        self.tracked_xp = tracked_xp - 1  # Fortran compensation
        self.tracked_yp = tracked_yp
        self.recreated = np.zeros((self.ts.par.profile_count,
                                   self.ts.par.profile_length))
        # self.weights = np.zeros((self.ts.par.profile_count,
        #                          tracked_xp.shape[0]))
        # self.bins = np.zeros((self.ts.par.profile_length,
        #                       self.ts.par.profile_count))

    # Flat profiles solution
    def run4(self):
        nparts = self.tracked_xp.shape[0]
        weights = np.zeros(nparts)
        flat_points = self.tracked_xp.copy()
        for i in range(self.ts.par.profile_count):
            flat_points[:, i] += self.ts.par.profile_length * i

        weights = self.calc_weigth_flattened(self.ts.profiles.flatten(),
                                             flat_points, weights, nparts)

        # diff_prof = None
        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            # t0 = tm.perf_counter()
            self.recreated = self.project_flattened(self.recreated.flatten(),
                                                    flat_points, weights,
                                                    nparts)
            # print(f'project: {tm.perf_counter() - t0}')

            # t0 = tm.perf_counter()
            self.recreated = self.recreated.reshape(self.ts.profiles.shape)
            # print(f'reshaping: {tm.perf_counter() - t0}')

            # t0 = tm.perf_counter()
            self.recreated = np.where(self.recreated < 0.0, 0.0, self.recreated)
            self.recreated /= np.sum(self.recreated, axis=1)[:, None]
            # print(f'normalizing and suppressing: {tm.perf_counter() - t0}')

            # t0 = tm.perf_counter()
            diff_prof = self.ts.profiles - self.recreated
            print(f'discrepancy: {self.discrepancy(diff_prof)}')
            # print(f'calculating discrepancy: {tm.perf_counter() - t0}')

            # self.analyze(0, diff_prof)

            # t0 = tm.perf_counter()
            weights = self.calc_weigth_flattened(diff_prof.flatten(),
                                                 flat_points, weights, nparts)
            # print(f'Back projecting: {tm.perf_counter() - t0}')

        # self.analyze(0, diff_prof)
        # plt.scatter(self.tracked_xp[:, 0], self.tracked_yp[:, 0], c=weights)
        # plt.show()

    @staticmethod
    @njit(parallel=True)
    def calc_weigth_flattened(flat_profiles, flat_points,
                              weights, nparts):
        for i in prange(nparts):
            weights[i] += np.sum(flat_profiles[flat_points[i]])
        return weights

    @staticmethod
    @njit(parallel=True)
    def project_flattened(flat_rec, flat_points, weights, nparts):
        for i in range(nparts):
            flat_rec[flat_points[i]] += weights[i]
        return flat_rec

    def run3(self):
        weights = np.zeros(self.tracked_xp.shape[0])

        self.calc_weights(weights, self.ts.profiles, self.tracked_xp)

        diff_prof = None
        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')
            self.project(weights, self.recreated,
                         self.tracked_xp, self.ts.par.profile_count)

            self.recreated = np.where(self.recreated < 0.0, 0.0, self.recreated)
            self.recreated /= np.sum(self.recreated, axis=1)[:, None]

            diff_prof = self.ts.profiles - self.recreated
            print(f'discrepancy: {self.discrepancy(diff_prof)}')

            # self.analyze(0, diff_prof)

            self.calc_weights(weights, diff_prof, self.tracked_xp)

        self.analyze(0, diff_prof)
        plt.scatter(self.tracked_xp[:, 0], self.tracked_yp[:, 0], c=weights)
        plt.show()

    @staticmethod
    @njit(parallel=True)
    def calc_weights(weights, profiles, points):
        for i in prange(len(weights)):
            for prof, po in enumerate(points[i]):
                weights[i] += profiles[prof, po]

    @staticmethod
    @njit(parallel=True)
    def _calc_weights(weights, profiles, points):
        
        nParts = points.shape[0]
        nProfs = points.shape[1]
#        print(points.shape)
#        sys.exit()
        for i in prange(nParts):
            pts = points[i]
#            print(pts)
#            for prof, pt in zip(profiles, pts):
            for j in range(nProfs):
                prof = profiles[j]
#                print(pt)
#                print(prof[pts])
                weights[i] += np.sum(prof[pts])
#                sys.exit()

    @staticmethod
    @njit(parallel=True)
    def project(weights, rec, points, profile_count):
        for p in prange(profile_count):
            for i, point in enumerate(points[:, p]):
                rec[p, point] += weights[i]


    def run(self):

        diff_prof = np.zeros(self.ts.profiles.shape)

        self.back_project_njit(self.tracked_xp,
                               self.ts.profiles,
                               self.weights,
                               self.ts.par.profile_count)

        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            t0 = tm.perf_counter()
            self.project_njit(self.tracked_xp, self.bins,
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
            self.back_project_njit(self.tracked_xp,
                                   diff_prof,
                                   self.bins,
                                   self.ts.par.profile_count)
            lg.info('back_projection_t: ' + str(tm.perf_counter() - t0))

        # TEMP
        # self.analyze(profilei=0, diff_prof=diff_prof)
        # END TEMP

    def run2(self):
        diff_prof = np.zeros(self.ts.profiles.shape)

        self.back_project_njit2(self.tracked_xp,
                                self.ts.profiles,
                                self.bins,
                                self.ts.par.profile_count)

        print('Iterating...')

        for i in range(self.ts.par.num_iter):
            print(f'iteration: {str(i + 1)} of {self.ts.par.num_iter}')

            t0 = tm.perf_counter()
            for p in range(self.ts.par.profile_count):
                diff_prof[p] = ((self.ts.profiles[p]
                                 / np.sum(self.ts.profiles[p]))
                                - (self.bins[:, p]
                                   / np.sum(self.bins[:, p])))
            print(f'difference time: {tm.perf_counter() - t0}')

            # TEMP
            # self.analyze(profilei=0, diff_prof=diff_prof)
            # END TEMP

            t0 = tm.perf_counter()
            self.back_project_njit2(self.tracked_xp,
                                    diff_prof,
                                    self.bins,
                                    self.ts.par.profile_count)
            print(f'projection time: {tm.perf_counter() - t0}')

            print(f'discrepancy: {self.discrepancy(diff_prof)}')

        # TEMP
        self.analyze(profilei=0, diff_prof=diff_prof)
        # END TEMP

    @staticmethod
    @njit(parallel=True)
    def project_njit(tracked_xp, bins, weights, profile_count):
        for profile in prange(profile_count):
            for point in range(tracked_xp.shape[0]):
                bins[tracked_xp[point, profile],
                     profile] += weights[profile, point]

    @staticmethod
    @njit(parallel=True)
    def back_project_njit(tracked_xp,
                          profiles,
                          weight_factors,
                          profile_count):

        for profile in prange(profile_count):
            counter = 0
            for point in tracked_xp[:, profile]:
                weight_factors[profile, counter] += profiles[profile, point]
                counter += 1

    @staticmethod
    @njit(parallel=True)
    def back_project_njit2(tracked_xp,
                           profiles,
                           bins,
                           profile_count):
        for profile in prange(profile_count):
            for index, point in enumerate(tracked_xp[:, profile]):
                bins[tracked_xp[index, profile], profile] += profiles[profile, point]

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
        plt.plot(self.recreated[profilei] / np.sum(self.recreated[profilei]))
        plt.gca().legend(('original', 'recreated python'))

        # Plotting difference
        plt.subplot(313)
        plt.title('Difference')
        plt.plot(diff_prof[profilei])
        plt.show()
    # END TEMP
