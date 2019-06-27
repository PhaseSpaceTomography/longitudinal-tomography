import numpy as np
import time as tm
import ctypes
import os
# from utils.assertions import TomoAssertions as ta
# from utils.exceptions import *
# import line_profiler
from physics import vrft
from numba import njit
from numpy.ctypeslib import ndpointer


# ====================
# HOW THE CLASS WORKS:
# ====================
#
# The (tomography) method is a hybrid one which incorporates particle tracking [1]. The reconstruction class
# is doing the particle tracking, which will make the basis for the tomography.
#
# A reconstruct object consists of one TimeSpace object which holds all the data from the measured profiles.
# The TimeSpace object also holds a parameters object with all the numbers needed for the reconstruction.
# In the reconstruction class can you also find a MapInfo object which is created based on the same parameters
# as the earlier mentioned TimeSpace object. The data from the MapInfo object tells the reconstruction algorithm
# which particles to track.
#
# The class starts from a homogeneous distribution of test particles at the profile to be reconstructed.
# The number of test particles is snpt^2 multiplyed by the number of active pixels from MapInfo.
# From this profile is the particles, with each their x and y coordinates, tracked using the 'longtrack' function
# to their position at the time of the following profile measurement ('dturns' turns later).
# The calculated x and y coordinates for the particles is then sent to the 'find_mapweight'.
# This function generates the data for the maps, mapsi and mapweights arrays.
#
# The maps array contains a index number for every 'submap'. In the first map will each submap contain snpt^2 particles.
# The index number of the submap is the index of a corresponding pair of vectors in the mapsi and mapsweight arrays.
# The mapsi array stores the xp coordinates as integers of the particles at a given time.
# The mapsweight array contains the number of particles at each of the associated mapsi indexes.
# In this way is it possible to follow the particles path from the original
# submap and through all the profile measurements. In the c++ accelerated version of the class is the
# mapsweights and mapsi arrays reduced to one dimensional for being fitted to the c++ routines.
#
# The reversed weights are calculated in the 'total_weightfactor' function. The reversedweights array
# holds the sum of tracked points at a certain x-point, casted to an iteger, divided by
# the total number of tracked points launched (and still within valid limits).
#
# sources:
#   1. S. Hancock, S. Koscielniak, M. Lindroos,
#       International Computational Accelerator Physics Conference, Darmstadt, Germany, 10 - 14 Sep 2000. Publ.
#       in: Physical Review Special Topics - Accelerators and Beams, vol. 3, 124202 (2000), (CERN-PS-2000-068-OP).
#
# ======================================
# MAP EXTENSION IN THE ORIGINAL PROGRAM:
# ======================================
#
# In the original fortran program was it implemented routines for extending the arrays mapsi, mapsweights and
#  reversedweights. This is not needed it you set the initial depth of maps (fmlistlength)
#  equal to the number of particles tracked in each cell (snpt^2).
#  The drawback of this method is the fact that the arrays bay be unnecessarily big and mostly
#  filled with default values. To read more about the old method, see the pure python implementation
#  of the reconstruction class: 'reconstruct.py'.
#
# =================
# GLOBAL VARIABLES:
# =================
# maps: 		    Array in three dimensions. The first refer to the projection. The two last to
#                    the i and j coordinates of the physical square area in which the picture
#                    will be reconstructed (submap). The map contains an integer
#                    which is is the index of the arrays, maps and mapsweight which in turn
#                    holds the actual data.
# mapsi: 		    Array in number of active points in maps and depth, mapsi holds the i in which
#                    mapsweight number of the orginally tracked Npt**2 number of points ended up.
# mapsweight:  	    Array in active points in maps and depth, see mapsi.
#
# reverseweight:    Array in profile_count and profile_length, holds the sum of tracked
#                    points at a certain i divided by the total number of tracked points
#                    launched (and still within valid limits).
# fmlistlength: 	(Initial) depth of maps.
#
# NB:
# All the arrays are reduced by one dimension to work optimally with the c++ functions during the tracking phase.
#    When the tracking is over, and the reversed weights are to be calculated, is the arrays reshaped to the above
#    mentioned shapes.
class ReconstructCpp:

    def __init__(self, timespace, mapinfo):
        self.timespace = timespace
        self.mapinfo = mapinfo
        self.maps = []
        self.mapsi = []
        self.mapweights = []
        self.reversedweights = []
        self.fmlistlength = timespace.par.snpt**2

        # Importing C++ functions
        lib_path = (os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])
                    + '/cpp_files/tomolib.so')
        tomolib = self._import_cfunctions(lib_path)

        self.find_mapweight = tomolib.weight_factor_array
        self.first_map = tomolib.first_map
        self.longtrack_cpp = tomolib.longtrack

    # @profile
    def run(self, film):
        tpar = self.timespace.par
        mi = self.mapinfo

        xpoints, ypoints = self._populate_bins(tpar.snpt)

        needed_maps = self._submaps_needed(tpar.filmstart,
                                           tpar.filmstop,
                                           tpar.filmstep,
                                           tpar.profile_count,
                                           mi.imin,
                                           mi.imax,
                                           mi.jmin,
                                           mi.jmax)

        print("Image" + str(film) + ": ")

        (maps, mapsi,
         mapsw) = self._init_arrays(tpar.profile_count,
                                    tpar.profile_length,
                                    needed_maps,
                                    tpar.snpt**2)

        nr_of_submaps = self.first_map(mi.jmin[film],
                                       mi.jmax[film],
                                       maps[film],
                                       mapsi,
                                       mapsw,
                                       mi.imin[film],
                                       mi.imax[film],
                                       tpar.snpt**2,
                                       tpar.profile_length)

        # Setting initial conditions for points to be tracked
        # xp and yp: time and energy in pixels
        # size: number of pixels pr profile
        xp = np.zeros(int(np.ceil(needed_maps * tpar.snpt**2
                                  / tpar.profile_count)))
        yp = np.zeros(int(np.ceil(needed_maps * tpar.snpt**2
                                  / tpar.profile_count)))

        direction = 1
        endprofile = tpar.profile_count
        submap_start = nr_of_submaps

        t0 = tm.time()
        for twice in range(2):

            turn_now = film * tpar.dturns

            (xp, yp,
             last_pxlidx) = self._init_tracked_point(tpar.snpt,
                                                     mi.imin[film],
                                                     mi.imax[film],
                                                     mi.jmin[film, :],
                                                     mi.jmax[film, :],
                                                     xp, yp,
                                                     xpoints,
                                                     ypoints)

            for profile in range(film + direction, endprofile, direction):

                if tpar.self_field_flag:
                    xp, yp, turn_now = self.longtrack_self(
                                            xp[:last_pxlidx],
                                            yp[:last_pxlidx],
                                            tpar.x_origin,
                                            tpar.h_num,
                                            tpar.omega_rev0,
                                            tpar.dtbin,
                                            tpar.phi0,
                                            tpar.yat0,
                                            mi.dEbin,
                                            turn_now,
                                            direction,
                                            tpar.dturns,
                                            tpar.dphase,
                                            tpar.phiwrap,
                                            self.timespace.vself,
                                            tpar.q,
                                            tpar.vrf1,
                                            tpar.vrf1dot,
                                            tpar.vrf2,
                                            tpar.vrf2dot,
                                            tpar.time_at_turn,
                                            tpar.h_ratio,
                                            tpar.phi12,
                                            tpar.deltaE0)
                else:
                    turn_now = self.longtrack_cpp(xp[:last_pxlidx],
                                                  yp[:last_pxlidx],
                                                  tpar.omega_rev0,
                                                  tpar.phi0,
                                                  tpar.dphase,
                                                  tpar.time_at_turn,
                                                  tpar.deltaE0,
                                                  np.int32(last_pxlidx),
                                                  tpar.x_origin,
                                                  tpar.dtbin,
                                                  mi.dEbin,
                                                  tpar.yat0,
                                                  tpar.phi12,
                                                  direction,
                                                  tpar.dturns,
                                                  turn_now,
                                                  tpar.q,
                                                  tpar.vrf1,
                                                  tpar.vrf1dot,
                                                  tpar.vrf2,
                                                  tpar.vrf2dot,
                                                  np.int32(tpar.h_num),
                                                  np.double(tpar.h_ratio))

                is_out = self.find_mapweight(xp,
                                             mi.jmin[film],
                                             mi.jmax[film],
                                             maps[profile],
                                             mapsi,
                                             mapsw,
                                             mi.imin[film],
                                             mi.imax[film],
                                             tpar.snpt ** 2,
                                             tpar.profile_length,
                                             tpar.snpt ** 2,
                                             submap_start)
                print(f'Tracking from time slice {str(film)} to '
                      f'{str(profile)} , {str(100 * is_out / last_pxlidx)}'
                      f'% went outside the image width. ')
                submap_start += nr_of_submaps

            direction = -1
            endprofile = -1
            print("mean iteration time: " +
                  str((tm.time() - t0) / tpar.profile_count))

            maps = maps.reshape((tpar.profile_count,
                                 tpar.profile_length,
                                 tpar.profile_length))
            mapsi = mapsi.reshape((needed_maps, tpar.snpt**2))
            mapsw = mapsw.reshape((needed_maps, tpar.snpt**2))

            reversedweights = self._total_weightfactor(
                                        tpar.profile_count,
                                        tpar.profile_length,
                                        tpar.snpt**2,
                                        tpar.snpt**2,
                                        mi.imin[film],
                                        mi.imax[film],
                                        mi.jmin[film, :],
                                        mi.jmax[film, :],
                                        mapsw,
                                        mapsi,
                                        maps)

            self.maps = maps
            self.mapsi = mapsi
            self.mapweights = mapsw
            self.reversedweights = reversedweights

    # Giving x and y coordinates to each particle in an arbitrary bin.
    def _populate_bins(self, snpt):
        xCoords = ((2.0 * np.arange(1, snpt + 1) - 1) / (2.0 * snpt))
        yCoords = xCoords

        xCoords = xCoords.repeat(snpt, 0).reshape((snpt, snpt))
        yCoords = np.repeat([yCoords], snpt, 0)
        return [xCoords, yCoords]

    # Calculating the total number of submaps needed for the reconstruction
    def _submaps_needed(self, filmstart, filmstop, filmstep,
                        profile_count, imin, imax, jmin, jmax):
        numaps = np.zeros(filmstop, dtype=int)
        for film in range(filmstart - 1, filmstop, filmstep):
            numaps[film] = np.sum(jmax[film,
                                       imin[film]:
                                       imax[film] + 1]
                                  - jmin[film,
                                         imin[film]:
                                         imax[film] + 1])
        return profile_count * int(np.amax(numaps))

    # Giving the arrays their initial shapes and values
    def _init_arrays(self, profile_count, profile_length,
                     nr_of_maps, array_depth):
        # All arrays are reduced by one dimension for use in c++ functions.
        #   When reconstruction is finished they wil have the following shapes:
        #    - maps [x, y, y]
        #    - mapsi [a, b]
        #    - mapsw [a, b]
        maps = np.zeros((profile_count, profile_length * profile_length),
                        dtype=np.int32)
        mapsi = np.full((nr_of_maps * array_depth), -1, dtype=np.int32)
        mapsweight = np.zeros((nr_of_maps * array_depth), dtype=np.int32)
        return maps, mapsi, mapsweight

    @staticmethod
    @njit
    # This function creates the arrays containing the
    # homogeneously distributed particles x and y coordinates.
    def _init_tracked_point(snpt, imin, imax,
                            jmin, jmax, xp, yp,
                            xpoints, ypoints):
        k = 0
        for iLim in range(imin, imax + 1):
            for jLim in range(jmin[iLim], jmax[iLim]):
                for i in range(snpt):
                    for j in range(snpt):
                        xp[k] = iLim + xpoints[i, j]
                        yp[k] = jLim + ypoints[i, j]
                        k += 1
        return xp, yp, k

    @staticmethod
    @njit(fastmath=True)
    # Particle tracking using self field voltages.
    def longtrack_self(xp, yp, xorigin, h_num,
                       omegarev0, dtbin, phi0, yat0,
                       debin, turn_now, direction, dturns,
                       dphase, phiwrap, vself, q,
                       vrf1, vrf1dot, vrf2, vrf2dot,
                       time_at_turn, hratio, phi12, deltae0):
        selfvolt = np.zeros(len(xp))
        dphi = ((xp + xorigin) * h_num * omegarev0[turn_now] * dtbin
                - phi0[turn_now])
        denergy = (yp - yat0) * debin
        if direction > 0:
            profile = int(turn_now / dturns)
            for i in range(dturns):
                dphi -= dphase[turn_now] * denergy
                turn_now += 1
                xp = (dphi + phi0[turn_now]
                      - xorigin * h_num * omegarev0[turn_now] * dtbin)
                xp = ((xp - phiwrap * np.floor(xp / phiwrap))
                      / (h_num * omegarev0[turn_now] * dtbin))
                # Much faster, but do not work with @njit:
                #   selfvolt = vself[profile, np.floor(xp).astype(int)]
                for j in range(len(xp)):
                    selfvolt[j] = vself[profile, int(np.floor(xp[j]))]
                denergy += q * (vrft(vrf1, vrf1dot, time_at_turn[turn_now])
                                * np.sin(dphi + phi0[turn_now])
                                + vrft(vrf2, vrf2dot, time_at_turn[turn_now])
                                * np.sin(hratio
                                         * (dphi + phi0[turn_now] - phi12))
                                + selfvolt
                                ) - deltae0[turn_now]
        else:
            profile = int(turn_now / dturns - 1)
            for i in range(dturns):
                # Much faster, but do not work with @njit:
                # selfvolt = vself[profile, np.floor(xp).astype(int)]
                for j in range(len(xp)):
                    selfvolt[j] = vself[profile, int(np.floor(xp[j]))]
                denergy -= q * (vrft(vrf1, vrf1dot, time_at_turn[turn_now])
                                * np.sin(dphi + phi0[turn_now])
                                + vrft(vrf2, vrf2dot, time_at_turn[turn_now])
                                * np.sin(hratio
                                         * (dphi + phi0[turn_now] - phi12))
                                + selfvolt
                                ) - deltae0[turn_now]
                turn_now -= 1
                dphi += dphase[turn_now] * denergy
                xp = (dphi + phi0[turn_now]
                      - xorigin * h_num * omegarev0[turn_now] * dtbin)
                xp = (xp - phiwrap * np.floor(xp / phiwrap)
                      / (h_num * omegarev0[turn_now] * dtbin))
        yp = denergy / debin + yat0
        return xp, yp, turn_now


    @staticmethod
    @njit
    # Going through submaps for calculating the total amount of points in each bin (x-coordinate as integer)
    def _total_weightfactor(profile_count, profile_length, fmlistlength,
                            npt, imin, imax, jmin, jmax,
                            mapsweight, mapsi, maps):
        reversedweights = np.zeros((profile_count, profile_length))
        for p in range(profile_count):
            for i in range(imin, imax + 1):
                for j in range(jmin[i], jmax[i]):
                    _reversedweights(reversedweights[p, :],         # out
                                     mapsweight[maps[p, i, j], :],  # in
                                     mapsi[maps[p, i, j], :],       # in
                                     fmlistlength,
                                     npt)
        return reversedweights * float(profile_count)

    @classmethod
    # Importing c++ functions and specifying input values.
    def _import_cfunctions(cls, lib_path):
        tomolib = ctypes.CDLL(lib_path)

        # Calculating the first map where all the particles are homogeneously distributed.
        tomolib.first_map.argtypes = [ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int]

        # Tracking the particles from the time of one profile measurement to the next.
        # Returns an array of the x- and y-coordinates of the current position of the particles.
        # Only the x-coordinates are needed for the calculations of the weight factors.
        # The longtrack function is basing its calculations of the next xp and yp arrays
        #    on its last output.
        tomolib.longtrack.argtypes = [ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_int,
                                      ctypes.c_double]

        # Calculating weight factors (mapsweights) and mapsi from tracked particles
        tomolib.weight_factor_array.argtypes = [ndpointer(ctypes.c_double),
                                                ndpointer(ctypes.c_int),
                                                ndpointer(ctypes.c_int),
                                                ndpointer(ctypes.c_int),
                                                ndpointer(ctypes.c_int),
                                                ndpointer(ctypes.c_int),
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_int]
        return tomolib

@njit
# Method for calculating the reversed weight factors for one submap
def _reversedweights(reversedweights, mapsweightarr,
                     mapsiarr, fmListLenght, npt):
    numpts = np.sum(mapsweightarr[:])
    if mapsiarr[fmListLenght - 1] < -1:
        raise NotImplementedError("Ext. maps not implemented.")
    for fl in range(npt):
        if fl < fmListLenght:
            if mapsiarr[fl] > 0:
                reversedweights[mapsiarr[fl]] += mapsweightarr[fl] \
                                               / float(numpts)
            else:
                break
        else:
            raise NotImplementedError("Ext. maps not implemented.")
