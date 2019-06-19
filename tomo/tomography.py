import numpy as np
import logging
import time as tm
from numba import njit
from utils.assertions import TomoAssertions as ta
from utils.exceptions import (ArrayLengthError,
                              ArgumentError,
                              PhaseSpaceReducedToZeroes)


class Tomography:

    def __init__(self, recontruction_data):
        self.rd = recontruction_data
        self.ts = self.rd.timespace
        self.mi = self.rd.mapinfo
        self.darray = None
        self.picture = None

        ta.assert_equal(self.rd.mapsi.shape, 'mapsi shape',
                        self.rd.mapweights.shape,
                        ArrayLengthError,
                        'mapsi should be of the same shape as mapsw')
        ta.assert_equal(self.rd.maps.shape, 'maps shape',
                        (self.ts.par.profile_count,
                         self.ts.par.profile_length,
                         self.ts.par.profile_length),
                        ArrayLengthError)


    def run(self, reconst_profile_idx):

        ta.assert_inrange(reconst_profile_idx,
                          'index of profile to reconstruct',
                          0, self.ts.par.filmstop, ArgumentError)

        darray = np.zeros(self.ts.par.num_iter + 1)
        phase_space = np.zeros((self.ts.par.profile_length,
                                self.ts.par.profile_length))

        self.backproject(self.ts.profiles,
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

        print("Iterating...")

        for i in range(self.ts.par.num_iter):
            logging.info("iteration #" + str(i + 1) + " of " + str(self.ts.par.num_iter))
            t = tm.process_time()
            # Project and find difference from last projection
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

            logging.info(f"Iteration lasted {tm.process_time() - t} seconds")

        # Calculate discrepancy for the last projection and write to file
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

    # Project phasespace onto profile_count profiles
    @staticmethod
    @njit
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
                                    += float(mapsweight[maps[p, i, j], fl]) \
                                       / numpts * picture[i, j]
                            else:
                                break
                        else:
                            raise NotImplementedError("[2]calculating "
                                                      + "projection with "
                                                      + "extended maps not "
                                                      + "yet implemented")
        return project

    # Backproject profile_count profiles onto
    # profile_length**2 phasespace
    @staticmethod
    @njit
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
                                    raise AssertionError('EXIT: Would have divided by zero in backproject.')
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

    # Calculate the discrepancy between projections and profiles
    @staticmethod
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
