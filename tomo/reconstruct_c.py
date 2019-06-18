import numpy as np
import time as tm
import ctypes
from utils.assertions import TomoAssertions as ta
from utils.exceptions import *
# import line_profiler
from physics import vrft
from numba import njit
from numpy.ctypeslib import ndpointer


class ReconstructCpp:

    def __init__(self, timespace, mapinfo):
        self.timespace = timespace
        self.mapinfo = mapinfo
        self.maps = []
        self.mapsi = []
        self.mapweights = []
        self.reversedweights = []
        self.fmlistlength = timespace.par.snpt**2

        ta.assert_equal(self.mapinfo.jmin.shape, 'jmin',
                        self.mapinfo.jmax.shape, ArrayLengthError,
                        'jmin and jmax should have the same shape')

        tomolib = ctypes.CDLL("/home/cgrindhe/tomo_v3/tomo/cpp_files/tomolib.so")

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
                                                ctypes.c_int]

        tomolib.first_map.argtypes = [ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int]

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

        self.find_mapweight = tomolib.weight_factor_array
        self.first_map = tomolib.first_map
        self.longtrack_cpp = tomolib.longtrack

    # @profile
    def reconstruct(self):
        tpar = self.timespace.par
        mi = self.mapinfo

        xpoints, ypoints = self._populate_bins(tpar.snpt)

        needed_maps = self._needed_amount_maps(tpar.filmstart,
                                               tpar.filmstop,
                                               tpar.filmstep,
                                               tpar.profile_count,
                                               mi.imin,
                                               mi.imax,
                                               mi.jmin,
                                               mi.jmax)

        for film in range(tpar.filmstart - 1, tpar.filmstop, tpar.filmstep):
            print("Image" + str(film) + ": ")

            # Initiating arrays.
            (maps, mapsi,
             mapsw) = self._init_arrays(tpar.profile_count,
                                        tpar.profile_length,
                                        needed_maps,
                                        tpar.snpt**2)

            # Calculating first map, indexes and weight factors.
            nr_of_submaps = self.first_map(mi.jmin[film],
                                           mi.jmax[film],
                                           maps[film],
                                           mapsi,
                                           mapsw,
                                           mi.imin[film],
                                           mi.imax[film],
                                           tpar.snpt**2,
                                           tpar.profile_length)

            # Set initial conditions for points to be tracked
            # xp and yp: time and energy in pixels
            # size: number of pixels pr profile

            xp = np.zeros(int(np.ceil(needed_maps * tpar.snpt**2
                                      / tpar.profile_count)))
            yp = np.zeros(int(np.ceil(needed_maps * tpar.snpt**2
                                      / tpar.profile_count)))

            direction = 1
            endprofile = tpar.profile_count
            start_submap = nr_of_submaps

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
                                                tpar.c1,
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
                                                      tpar.c1,
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
                                                 start_submap)
                    print(str(profile) + ": " + str(is_out))
                    start_submap += nr_of_submaps

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

    def _needed_amount_maps(self, filmstart, filmstop, filmstep,
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

    def _populate_bins(self, snpt):
        xCoords = ((2.0 * np.arange(1, snpt + 1) - 1)
                   / (2.0 * snpt))
        yCoords = xCoords

        xCoords = xCoords.repeat(snpt, 0).reshape(
            (snpt, snpt))
        yCoords = np.repeat([yCoords], snpt, 0)
        return [xCoords, yCoords]

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
    def longtrack_self(xp, yp, xorigin, h_num,
                       omegarev0, dtbin, phi0, yat0,
                       debin, turn_now, direction, dturns,
                       c1, phiwrap, vself, q,
                       vrf1, vrf1dot, vrf2, vrf2dot,
                       time_at_turn, hratio, phi12, deltae0):
        selfvolt = np.zeros(len(xp))
        dphi = ((xp + xorigin) * h_num * omegarev0[turn_now] * dtbin
                - phi0[turn_now])
        denergy = (yp - yat0) * debin
        if direction > 0:
            profile = int(turn_now / dturns)
            for i in range(dturns):
                dphi -= c1[turn_now] * denergy
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
                dphi += c1[turn_now] * denergy
                xp = (dphi + phi0[turn_now]
                      - xorigin * h_num * omegarev0[turn_now] * dtbin)
                xp = (xp - phiwrap * np.floor(xp / phiwrap)
                      / (h_num * omegarev0[turn_now] * dtbin))
        yp = denergy / debin + yat0
        return xp, yp, turn_now


    @staticmethod
    @njit
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

@njit
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
