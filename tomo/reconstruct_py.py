import numpy as np
import logging
import time as tm
# from line_profiler import LineProfiler
from numba import njit
from physics import vrft


# SOME VARIABLES USED IN THIS CLASS:
# ----------------------------------
#
# maps: 		 	Array in three dimensions. The first refer to the projection. The two last to
#                 the i and j coordinates of the physical square area in which the picture
#                 will be reconstructed. The map contains an integer
#                 which is is the index of the arrays, maps and mapsweight which in turn
#                 holds the actual data.
# mapsi: 		 	Array in number of active points in maps and depth, mapsi holds the i in which
#                 mapsweight number of the orginally tracked Npt**2 number of points ended up.
# mapsweight:  	Array in active points in maps and depth, see mapsi
#
# reverseweight: 	Array in profile_count and profile_length, holds the sum of tracked
#                 points at a certain i divided by the total number of tracked points
#                 launched (and still within valid limits)
# fmlistlength: 	Initial depth of maps
#
# SOME VARIABLES USED IN THE ORIGINAL FORTRAN VERSION
# BUT NOT IN THE PYTHON VERSION - MAP EXTENSION:
# ----------------------------------------------------
# mapsweightx: 	Array in number of extended mapsweight vectors and depth, holds the
#                 overflow from mapsweight
# mapsix: 	 	Array in number of extended mapsi vectors and depth, mapsix holds the overflow
#                 from mapsi (last element dimension depth in mapsi holds the first index for
#                 mapsix
# xlength: 		The number of extended vectors in maps
# xunit: 			The number of vectors additionally allocated every time a new allocation
#                 takes place
#
# MAP EXTENSIONS
# ----------------
# In the original program the calculation of the depth of the maps, fmlistlength,
#  was calculated in the following way (see function _depth_maps):
#         arg1 = max(4, np.floor(0.1 * profile_length))
#         arg2 = min(snpt**2, profile_length)
#         fmlistlength = int(min(arg1, arg2))
# In normal cases the profile length is much larger than the number of tracked points
#  in a cell. In these cases the fmlistlength, which is the number of elements in the
#  mapsi and mapsweight function, are set to the number of tracked points in each cell.
#  In cases where the number of tracked points are very large and/or the profile length
#  are very small, there is a chance that fmlistlength will be set to another smaller number.
#  This is in many cases unproblematic but not for all.
# In the function '_calc_weightfactors', there is loaded a vector 'xvec' into the
#  function '_calc_one_weightfactor'. If the square root of the number of tracked points (snpt)
#  equals three, then xvec could be something like [2, 2, 2, 3, 3, 2, 2, 4, 4]
#  In the first iteration through the array the corresponding xet = xvec[0] = 2.
#  The corresponding xnum would then be [1, 1, 1, 0, 0, 1, 1, 0, 0].
#  The sum of this vector (in this case 5) would be the mapweight[0]
#  The integer, with is a rounded down x-coordinate is saved in mapsi[0]
#  The next 'xet' would in this case be 3, and so on.
#  The calculate_one_weightfactor contiues in this way until
#   there are no more unused numbers in the xvec.
#  After saving the xet to mapsi[0] and the sum of the xnum vector to mapsweight[0]
#   an counter is raised with the value of one, pointing at the next element in both arrays.
# In cases where fmlistlength = snpt**2, the length of the mapsi and mapsweight
#  arrays will be snpt**2. Since the xvec always has the size snpt**2, there can never be
#   an overflow of the mapsi and mapsweight arrays. A new example:
#   If the xvec array is [1, 2, 3, 4, 5, 6, 7, 8, 9] then the final
#   mapsi in this case be [1, 2, 3, 4, 5, 6, 7, 8, 9]
#   and mapweight [1, 1, 1, 1, 1, 1 ,1, 1, 1].
# The example over is a case which would be a problem is fmlistlength is sat to
#  less than snpt**2. If for some reason fmlistlength would be in this case set to 6,
#  then mapsi and mapsweight would have length (in axis=1) to be 6.
#  This would result in an overflow, and a need for a map extension of 3 elements.
# The positive thing with such a solution is the saving of memory.
#  Since the mapsi and mapsweight arrays is initiated with the shape (number of maps, fmlistlength)
#  there is large potential to save some memory if only the two-three elements in a very long
#  array is being used. Mabye the mapsi array will be initiated as shape (1 000 000, 16), when only
#  (1 000 000, 8) would be needed.
# However, at this time i find this method unnecessarily complex
# in relation to the gain achieved from it. My solution
#  is to set the fmlistlength = snpt**2. In this way the arrays will never overflow.
#  points in which the map extension was being used is marked with
#   'raise NotImplementedError("Ext. maps not implemented.")' in this class and the tomography class.
class Reconstruct:

    def __init__(self, timespace, mapinfo):
        logging.info("Running reconstruct_py")
        self.timespace = timespace
        self.mapinfo = mapinfo
        self.maps = []
        self.mapsi = []
        self.mapweights = []
        self.reversedweights = []
        self.fmlistlength = 0

    # @profile
    def run(self, film):
        mi = self.mapinfo
        ts = self.timespace
        tpar = self.timespace.par
        npt = tpar.snpt**2

        (points, nr_of_maps,
         fmlistlength) = self._init_reconstruction(ts, mi)

        xpoints = points[0]
        ypoints = points[1]

        print("Image" + str(film) + ": ")

        # Initiating arrays
        (maps, mapsi,
         mapsweight) = self._init_arrays(tpar.profile_count,
                                         tpar.profile_length,
                                         nr_of_maps,
                                         fmlistlength)

        # Do the first map
        (maps[film, :, :], mapsi,
         mapsweight, actmaps) = self._first_map(
                                        mi.imin[film],
                                        mi.imax[film],
                                        mi.jmin[film, :],
                                        mi.jmax[film, :],
                                        npt,
                                        maps[film, :, :],
                                        mapsi,
                                        mapsweight)

        # Set initial conditions for points to be tracked
        # xp and yp: time and energy in pixels
        # size: number of pixels pr profile

        xp = np.zeros(int(np.ceil(nr_of_maps * npt
                                  / tpar.profile_count)))
        yp = np.zeros(int(np.ceil(nr_of_maps * npt
                                  / tpar.profile_count)))

        direction = 1
        endprofile = tpar.profile_count
        t0 = tm.time()
        for twice in range(2):

            turn_now = film * tpar.dturns

            (xp, yp,
             last_pxlidx) = Reconstruct._init_tracked_point(
                                            tpar.snpt,
                                            mi.imin[film],
                                            mi.imax[film],
                                            mi.jmin[film, :],
                                            mi.jmax[film, :],
                                            xp, yp, xpoints, ypoints)

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
                                            ts.vself,
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
                    xp, yp, turn_now = self.longtrack(
                                            direction,
                                            tpar.dturns,
                                            yp[:last_pxlidx],
                                            xp[:last_pxlidx],
                                            mi.dEbin,
                                            turn_now,
                                            tpar.x_origin,
                                            tpar.h_num,
                                            tpar.omega_rev0,
                                            tpar.dtbin,
                                            tpar.phi0,
                                            tpar.yat0,
                                            tpar.dphase,
                                            tpar.deltaE0,
                                            tpar.vrf1,
                                            tpar.vrf1dot,
                                            tpar.vrf2,
                                            tpar.vrf2dot,
                                            tpar.time_at_turn,
                                            tpar.h_ratio,
                                            tpar.phi12,
                                            tpar.q)

                # Calculating weight factors for each map
                # Extends maps if needed
                isOut, actmaps = self._calc_weightfactors(
                                        mi.imin[film],
                                        mi.imax[film],
                                        mi.jmin[film, :],
                                        mi.jmax[film, :],
                                        maps[profile, :, :],
                                        mapsi[:, :],
                                        mapsweight[:, :],
                                        xp,
                                        npt,
                                        tpar.profile_length,
                                        fmlistlength,
                                        actmaps)

                # TEMP
                self._new_calc_wf(xp)
                raise SystemExit
                # END TEMP

                print(f"Tracking from time slice { str(film) } to "
                      f"{str(profile)} , {str(100 * isOut / last_pxlidx)}"
                      f"% went outside the image width. ")

            direction = -1
            endprofile = -1

            print("Mean iteration time: "
                  + str((tm.time() - t0) / tpar.profile_count))

            logging.info("Calculating reversed weight")
            t = tm.process_time()
            reversedweights = self._total_weightfactor(
                                        tpar.profile_count,
                                        tpar.profile_length,
                                        fmlistlength, npt,
                                        mi.imin[film],
                                        mi.imax[film],
                                        mi.jmin[film, :],
                                        mi.jmax[film, :],
                                        mapsweight, mapsi, maps)

            logging.info(f"reversed weights calculated in {tm.process_time() - t} sek")

        self.maps = maps
        self.mapsi = mapsi
        self.mapweights = mapsweight
        self.reversedweights = reversedweights
        self.fmlistlength = fmlistlength

    # This functions sets up parameters for maps
    #   based on mapInfo and timespace object
    def _init_reconstruction(self, ts, mi):
        points = self._populate_bins(ts.par.snpt)

        nr_of_maps = self._needed_amount_maps(
                            ts.par.filmstart,
                            ts.par.filmstop,
                            ts.par.filmstep,
                            ts.par.profile_count, mi)

        # Variables needed for extendig of maps
        # fmlistlength = self._depth_maps(
        #                    ts.par.snpt,
        #                    ts.par.profile_length)
        # xLength = int(np.ceil(0.1*nr_of_maps))
        # xUnit = xLength

        fmlistlength = ts.par.snpt**2



        return points, nr_of_maps, fmlistlength

    def _populate_bins(self, sqrtNbrPoints):
        xCoords = ((2.0 * np.arange(1, sqrtNbrPoints + 1) - 1)
                   / (2.0 * sqrtNbrPoints))
        yCoords = xCoords

        xCoords = xCoords.repeat(sqrtNbrPoints, 0).reshape(
            (sqrtNbrPoints, sqrtNbrPoints))
        yCoords = np.repeat([yCoords], sqrtNbrPoints, 0)
        return [xCoords, yCoords]

    def _init_arrays(self, profile_count, profile_length,
                     nr_of_maps, fmlistlength):
        maps = np.zeros((profile_count, profile_length, profile_length), int)
        mapsi = np.full((nr_of_maps, fmlistlength), -1, int)
        mapsweight = np.zeros((nr_of_maps, fmlistlength), int)
        return maps, mapsi, mapsweight

    def _needed_amount_maps(self, filmstart, filmstop, filmstep,
                            profile_count, mi):
        numaps = np.zeros(filmstop, dtype=int)
        for profile in range(filmstart - 1, filmstop, filmstep):
            numaps[profile] = np.sum(mi.jmax[profile,
                                             mi.imin[profile]:
                                             mi.imax[profile] + 1]
                                     - mi.jmin[profile,
                                               mi.imin[profile]:
                                               mi.imax[profile] + 1])
        return profile_count * int(np.amax(numaps))

    def _depth_maps(self, snpt, profile_length):
        arg1 = max(4, np.floor(0.1 * profile_length))
        arg2 = min(snpt**2, profile_length)
        return int(min(arg1, arg2))

    @staticmethod
    @njit
    def _first_map(imin, imax, jmin, jmax, npt,
                   maps, mapsi, mapsweight):
        actmaps = 0
        for i in range(imin, imax + 1):
            for j in range(jmin[i], jmax[i]):
                maps[i, j] = actmaps
                mapsi[actmaps, 0] = i
                actmaps += 1
        mapsweight[0:actmaps, 0] = npt
        return maps, mapsi, mapsweight, actmaps

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
    @njit
    def longtrack(direction, nrep, yp, xp, dEbin, turn_now, x_origin,
                  h_num, omega_rev0, dtbin, phi0, yat0, dphase, deltaE0,
                  vrf1, vrf1dot, vrf2, vrf2dot, time_at_turn,
                  h_ratio, phi12, q):

        dphi = ((xp + x_origin) * h_num * omega_rev0[turn_now] * dtbin
                - phi0[turn_now])
        denergy = (yp - yat0) * dEbin

        if direction > 0:
            for i in range(nrep):
                dphi -= dphase[turn_now] * denergy
                turn_now += 1
                denergy += q * (vrft(vrf1, vrf1dot, time_at_turn[turn_now])
                                * np.sin(dphi + phi0[turn_now])
                                + vrft(vrf2, vrf2dot, time_at_turn[turn_now])
                                * np.sin(h_ratio
                                         * (dphi + phi0[turn_now] - phi12))
                                ) - deltaE0[turn_now]
        else:
            for i in range(nrep):
                denergy -= q * (vrft(vrf1, vrf1dot, time_at_turn[turn_now])
                                * np.sin(dphi + phi0[turn_now])
                                + vrft(vrf2, vrf2dot, time_at_turn[turn_now])
                                * np.sin(h_ratio
                                         * (dphi + phi0[turn_now] - phi12))
                                ) - deltaE0[turn_now]
                turn_now -= 1
                dphi += dphase[turn_now] * denergy
        xp = ((dphi + phi0[turn_now])
              / (float(h_num) * omega_rev0[turn_now] * dtbin)
              - x_origin)
        yp = denergy / float(dEbin) + yat0

        return xp, yp, turn_now

    @staticmethod
    @njit
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

    def _new_calc_wf(self, inn_xp):
        particles_pr_pxl = self.timespace.par.snpt**2
        nr_pixels = np.int(len(inn_xp) / self.timespace.par.snpt**2)

        xp = np.ceil(inn_xp.copy()).astype(int)
        xp = xp.reshape((np.int(len(xp) / self.timespace.par.snpt**2), self.timespace.par.snpt**2))

        bin = np.zeros(np.max(xp) + 1)
        for coordinate in xp[0]:
            bin[coordinate] += 1

        indices = np.zeros(particles_pr_pxl)
        weights = np.zeros(particles_pr_pxl)

        j = 0
        for i in range(len(bin)):
            if bin[i] != 0:
                indices[j] = i
                weights[j] = bin[i]
                j += 1
        print(indices)
        print(weights)


        indices = np.zeros(1000000, dtype=int)
        weights = np.zeros(1000000, dtype=int)

        # j = 0
        # for i in range(n_bins):
        #     if bins[i] != 0:
        #         indices[j] = i
        #         weights[j] = bins[i]
        #         j += 1
        # indices = indices[:j]
        # weights = weights[:j]
        # pass


    @staticmethod
    @njit
    def _calc_weightfactors(imin, imax, jmin,
                            jmax, maps, mapsi,
                            mapsweight, xp, npt,
                            profile_length, fmlistlength, actmaps):
        ioffset = 0
        isout = 0
        uplim = 0

        xlog = np.array([True] * npt)
        xnumb = np.zeros(npt)

        for i in range(imin, imax + 1):
            for j in range(jmin[i], jmax[i]):
                maps[i, j] = actmaps
                lowlim = (j - jmin[i]) * npt + ioffset
                uplim = (j - jmin[i] + 1) * npt + ioffset
                xp_segment = np.ceil(xp[lowlim: uplim])

                isout += _calc_one_weightfactor(
                                            npt,
                                            mapsi[maps[i, j], :],
                                            mapsweight[maps[i, j], :],
                                            profile_length,
                                            fmlistlength,
                                            xp_segment,
                                            xlog, xnumb)
                actmaps += 1
            ioffset = uplim
        return isout, actmaps

    @staticmethod
    @njit
    def _total_weightfactor(profile_count, profile_length, fmlistlength,
                            npt, imin, imax, jmin, jmax,
                            mapsweight, mapsi, maps):
        reversedweights = np.zeros((profile_count, profile_length))
        for p in range(profile_count):
            for i in range(imin, imax + 1):
                for j in range(jmin[i], jmax[i]):
                    _reversedweights(reversedweights[p, :],      # out
                                  mapsweight[maps[p, i, j], :],  # in
                                  mapsi[maps[p, i, j], :],       # in
                                      fmlistlength,
                                      npt)
        return reversedweights * float(profile_count)


@njit
def _calc_one_weightfactor(npt, mapsi, mapsweight, profile_length,
                           fmlistlength, xvec, xlog, xnumb):
    isout = 0
    icount = 0
    xlog[:] = True
    for l in range(npt):
        if xlog[l]:
            xet = xvec[l]
            xnumb[np.logical_and(xvec == xet, xlog)] = 1
            xlog[np.logical_and(xvec == xet, xlog)] = 0
            if xet < 1 or xet > profile_length:
                isout += 1
            else:
                if icount < fmlistlength:
                    mapsi[icount] = int(xet) - 1
                    mapsweight[icount] = int(np.sum(xnumb))
                    xnumb[:] = 0
                else:
                    raise NotImplementedError("Ext. maps not implemented.")
                icount += 1
    return isout

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
