import matplotlib.pyplot as plt
import numpy as np
from numba import njit


class NewTomography:

    def __init__(self, timespace, tracked_points):
        self.ts = timespace
        self.tracked_points = tracked_points - 1  # Fortran
        self.weights = np.zeros(tracked_points.shape[0])
        self.bins = np.zeros((self.ts.par.profile_length,
                              self.ts.par.profile_count))

    def run(self):

        picture = 0
        print(f'Back projecting profile: {picture}')
        self.back_project_njit(self.tracked_points, self.ts.profiles,
                               self.weights, picture)

        for i in range(self.ts.par.num_iter):
            print(f'Projecting profile: {picture}')
            self.project_njit(self.tracked_points, self.bins,
                              self.weights, picture)
            self.weights[:] = 0

            diff_prof = ((self.ts.profiles[picture]
                          / np.sum(self.ts.profiles[picture]))
                         - (self.bins[:, picture]
                            / np.sum(self.bins[:, picture])))

            # TEMP
            # Plotting projection
            plt.figure()
            plt.subplot(311)
            # plt.title('Reconstructed')
            plt.plot(self.bins[:, 0] / np.sum(self.bins[:, 0]))
            # plt.subplot(312)
            # plt.title('Original')
            plt.plot(self.ts.profiles[0] / np.sum(self.ts.profiles[0]))
            plt.subplot(313)
            plt.title('Difference')
            plt.plot(diff_prof)
            plt.show()
            # END TEMP

            # TEMP
            diff_prof = np.concatenate((diff_prof, diff_prof))
            diff_prof = diff_prof.reshape((2, 205))
            print(diff_prof.shape)
            # END TEMP

            self.back_project_njit(self.tracked_points, diff_prof,
                                   self.weights, picture)






        diff_prof = self.ts.profiles[0] - self.bins[:, 0]

        # TEMP
        # Plotting difference
        # plt.plot(diff_prof)
        # plt.show()
        # END TEMP

    def project(self, profile_idx):
        for p in range(self.tracked_points.shape[0]):
            self.bins[self.tracked_points[p, profile_idx],
                      profile_idx] += self.weights[p]

    @staticmethod
    @njit
    def project_njit(tracked_points, bins, weights, profile_idx):
        for p in range(tracked_points.shape[0]):
            bins[tracked_points[p, profile_idx], profile_idx] += weights[p]

    def back_project(self, profile_idx):
        one_profile_points = self.tracked_points[:, 0]
        one_profile = self.ts.profiles[profile_idx, :]
        counter = 0
        for point in one_profile_points:
            self.weights[counter] += one_profile[point]
            counter += 1

    @staticmethod
    @njit
    def back_project_njit(tracked_points, profiles,
                          weights, profile_idx):
        one_profile_points = tracked_points[:, 0]
        one_profile = profiles[profile_idx, :]
        counter = 0
        for point in one_profile_points:
            weights[counter] += one_profile[point]
            counter += 1
