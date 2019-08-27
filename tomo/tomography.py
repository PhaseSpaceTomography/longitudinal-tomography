import numpy as np
import logging
import time as tm
from numba import njit
from utils.assertions import TomoAssertions as ta
from utils.exceptions import (ArrayLengthError,
                              ArgumentError,
                              PhaseSpaceReducedToZeroes)

# ===============
# ABOUT THE CLASS
# ===============
#
# The tomography class is using the ART algorithm.
# to perform tomography based on the profiles from a timeSpace object
# and the particle tracking from a Reconstruction object.
#
# ========================
# VARIABLES FOR THIS CLASS
# ========================
#
# darray        Array containing the difference between projections and actual measured profile
# phase_space   Phase space to be reconstructed. Saved in object as picture as the final output of the program.
#

class Tomography:

    def __init__(self, recontruction_data):
        self.rd = recontruction_data
        self.ts = self.rd.timespace
        self.mi = self.rd.mapinfo
        self.darray = None
        self.picture = None

    # The function to be run. Going through the iterative process of the tomography
    # algorithm. Returns the final recreated picture 'picture' and a history of
    # the convergence of the recreate phase space in comparison to the actual profile.
    def run(self, reconst_profile_idx):

        self.validate_reconstruction(reconst_profile_idx)

        darray, phase_space = self.init_arrays()

        self.backproject(self.ts.profiles,
                         phase_space,
                         self.mi.imin,
                         self.mi.imax,
                         self.mi.jmin,
                         self.mi.jmax,
                         self.rd.maps,
                         self.rd.mapsi,
                         self.rd.mapweights,
                         self.rd.reversedweights,
                         self.rd.fmlistlength,
                         self.ts.par.profile_count,
                         self.ts.par.snpt)

        print("Iterating...")
        for i in range(self.ts.par.num_iter):
            print(f'iteration #{str(i + 1)} of {str(self.ts.par.num_iter)}')
            diffprofiles = (self.ts.profiles
                            - self.project(phase_space,
                                           self.mi.imin[reconst_profile_idx],
                                           self.mi.imax[reconst_profile_idx],
                                           self.mi.jmin[reconst_profile_idx],
                                           self.mi.jmax[reconst_profile_idx],
                                           self.rd.maps,
                                           self.rd.mapsi,
                                           self.rd.mapweights,
                                           self.rd.fmlistlength,
                                           self.ts.par.snpt,
                                           self.ts.par.profile_count,
                                           self.ts.par.profile_length))

            darray[i] = self.discrepancy(diffprofiles,
                                         self.ts.par.profile_length,
                                         self.ts.par.profile_count)

            self.backproject(diffprofiles,
                             phase_space,
                             self.mi.imin[reconst_profile_idx],
                             self.mi.imax[reconst_profile_idx],
                             self.mi.jmin[reconst_profile_idx],
                             self.mi.jmax[reconst_profile_idx],
                             self.rd.maps,
                             self.rd.mapsi,
                             self.rd.mapweights,
                             self.rd.reversedweights,
                             self.rd.fmlistlength,
                             self.ts.par.profile_count,
                             self.ts.par.snpt)

            phase_space = self.supress_zeroes_and_normalize(phase_space)

        # Calculate discrepancy for the last projection, before returning final picture.
        diffprofiles = (self.ts.profiles
                        - self.project(phase_space,
                                       self.mi.imin[reconst_profile_idx],
                                       self.mi.imax[reconst_profile_idx],
                                       self.mi.jmin[reconst_profile_idx],
                                       self.mi.jmax[reconst_profile_idx],
                                       self.rd.maps,
                                       self.rd.mapsi,
                                       self.rd.mapweights,
                                       self.rd.fmlistlength,
                                       self.ts.par.snpt,
                                       self.ts.par.profile_count,
                                       self.ts.par.profile_length))

        darray[self.ts.par.num_iter] = self.discrepancy(
                                            diffprofiles,
                                            self.ts.par.profile_length,
                                            self.ts.par.profile_count)
        return darray, phase_space

    @staticmethod
    @njit
    # Projecting back to the one dimensional arrays in order to compare
    # reconstructed phase space to the original measured profile.
    def project(picture,
                imin, imax, jmin, jmax,
                maps, mapsi, mapsweight,
                fmlistlength, snpt,
                profile_count, profile_length):
        project = np.zeros((profile_count, profile_length))
        for p in range(profile_count):
            for i in range(imin, imax + 1):
                for j in range(jmin[i], jmax[i]):
                    numpts = float(np.sum(mapsweight[maps[p, i, j], :]))
                    if mapsi[maps[p, i, j], fmlistlength - 1] < -1:
                        raise NotImplementedError("[1]calculating "
                                                  + "projection with "
                                                  + "extended maps not "
                                                  + "yet implemented")
                    for fl in range(snpt**2):
                        if fl < fmlistlength:
                            if mapsi[maps[p, i, j], fl] > 0:
                                project[p, mapsi[maps[p, i, j], fl]] \
                                    += (float(mapsweight[maps[p, i, j], fl])
                                        / numpts
                                        * picture[i, j])
                            else:
                                break
                        else:
                            raise NotImplementedError("[2]calculating "
                                                      + "projection with "
                                                      + "extended maps not "
                                                      + "yet implemented")
        return project


    @staticmethod
    @njit
    #  Back projecting all bins of all profiles
    #  to give an approximation to the original distribution.
    def backproject(profiles, back_proj,
                    imin, imax, jmin, jmax,
                    maps, mapsi, mapsweight,
                    reversedweight, fmlistlength,
                    profile_count,
                    snpt):
        for p in range(profile_count):
            for i in range(imin, imax + 1):
                for j in range(jmin[i], jmax[i]):
                    numpts = float(np.sum(mapsweight[maps[p, i, j], :]))
                    if mapsi[maps[p, i, j], fmlistlength - 1] < -1:
                        raise NotImplementedError("[1]calculating backproject with"
                                                  + "extended maps not "
                                                  + "yet implemented")
                    for fl in range(snpt**2):
                        if fl < fmlistlength:
                            if mapsi[maps[p, i, j], fl] > 0:
                                if reversedweight[p, mapsi[maps[p, i, j], fl]] <= 0:
                                    raise AssertionError('EXIT: Would have '
                                                         'divided by zero '
                                                         'in backproject.')
                                back_proj[i, j] \
                                    += (float(mapsweight[maps[p, i, j], fl])
                                        / float(numpts)
                                        / reversedweight[p, mapsi[maps[p, i, j], fl]]
                                        * profiles[p, mapsi[maps[p, i, j], fl]])
                            else:
                                break
                        else:
                            raise NotImplementedError("[2]calculating backproject with"
                                                      + "extended maps not "
                                                      + "yet implemented")


    @staticmethod
    # Calculate the discrepancy between projected and measured profiles.
    def discrepancy(diffprofiles, profile_length, profile_count):
        return np.sqrt(np.sum(diffprofiles**2)/(profile_length*profile_count))

    @staticmethod
    def supress_zeroes_and_normalize(phase_space):
        phase_space = np.where(phase_space < 0.0, 0.0, phase_space)
        if np.all(phase_space <= 0.0):
            raise PhaseSpaceReducedToZeroes("Phase space reduced to zeroes!")
        else:
            phase_space /= float(np.sum(phase_space))
        return phase_space

    # Validating input from reconstruction
    def validate_reconstruction(self, rec_idx):
        ta.assert_equal(self.rd.mapsi.shape, 'mapsi shape',
                        self.rd.mapweights.shape,
                        ArrayLengthError,
                        'mapsi should be of the same shape as mapsw')
        ta.assert_equal(self.rd.maps.shape, 'maps shape',
                        (self.ts.par.profile_count,
                         self.ts.par.profile_length,
                         self.ts.par.profile_length),
                        ArrayLengthError)
        ta.assert_inrange(rec_idx,
                          'index of profile to reconstruct',
                          0, self.ts.par.filmstop, ArgumentError)

    def init_arrays(self):
        darray = np.zeros(self.ts.par.num_iter + 1)
        phase_space = np.zeros((self.ts.par.profile_length,
                                self.ts.par.profile_length))
        return darray, phase_space

    def out_darray_txtfile(self, file_path, film_idx):
        full_file_path = (file_path + "py_d" + str(film_idx + 1) + ".dat")
        with open(full_file_path, "w") as file:
            for i in range(self.darray.size):
                file.write(str(i) + '\t' + str(self.darray[i]) + '\n')
            file.close()
        logging.info("array saved to: " + full_file_path)

    def out_picture(self, file_path, film):
        full_file_path = (file_path + "py_picture" + str(film + 1) + ".dat")
        np.savetxt(full_file_path, self.picture.reshape((self.ts.par.profile_length**2)))
        logging.info("Picture saved to " + full_file_path)
