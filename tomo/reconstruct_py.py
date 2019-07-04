import numpy as np
import logging
import time as tm
from multiprocessing import Pool, current_process
# from line_profiler import LineProfiler
from numba import njit, prange
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

        # TEMP?
        self.film = None
        self.all_turns = (timespace.par.dturns
                          * (timespace.par.profile_count - 1))
        # END TEMP

    def new_run(self, film):
        self.film = film
        mi = self.mapinfo
        ts = self.timespace
        tpar = self.timespace.par

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
            tpar.snpt**2,
            maps[film, :, :],
            mapsi,
            mapsweight)

        # Set initial conditions for points to be tracked
        # xp and yp: time and energy in pixels
        # size: number of pixels pr profile

        # xp = np.zeros(int(np.ceil(nr_of_maps * tpar.snpt**2
        #                           / tpar.profile_count)))
        # yp = np.zeros(int(np.ceil(nr_of_maps * tpar.snpt**2
        #                           / tpar.profile_count)))

        initial_points = np.zeros((2, int(np.ceil(nr_of_maps * tpar.snpt**2
                                   / tpar.profile_count))))

        # Creating the first profile with equally distributed points
        (initial_points[0],
         initial_points[1],
         last_pxlidx) = Reconstruct._init_tracked_point(
                                        tpar.snpt,
                                        mi.imin[film],
                                        mi.imax[film],
                                        mi.jmin[film, :],
                                        mi.jmax[film, :],
                                        initial_points[0],
                                        initial_points[1],
                                        xpoints, ypoints)

        initial_points

        # LONGTRACK
        # -----------------------
        # TEMP
        rec_prof = 0
        # END TEMP

        t = tm.perf_counter()
        # for pp in initial_points:
        #     _ = self.longtrack_one_particle_mod(pp)
        pool = Pool()

        initial_params = np.zeros((2, int(np.ceil(nr_of_maps * tpar.snpt ** 2
                                  / tpar.profile_count))))
        (initial_params[0],
         initial_params[1]) = self.calc_dphi_denergy(initial_points[0],
                                                     initial_points[1],
                                                     rec_prof)
        initial_params = initial_params.T

        pool.map(self.longtrack_one_particle_mod, initial_params)

        # self.longtrack_one_particle_mod(initial_params[0])

        # all_xp = self.longtrack_all(xp, yp, film)
        # check = all_xp[:, 0]
        # del all_xp
        # diff = one_xp - check
        # print(f'Accumulated difference algorithms: {np.sum(np.abs(diff))}')
        print('Longtrack time: ' + str(tm.perf_counter() - t))
        raise SystemExit
        del xp
        # -----------------------
        # CASTING ALL XP
        # -----------------------
        t = tm.perf_counter()
        all_xp = np.ceil(all_xp).astype(int)
        print('Casting time: ' + str(tm.perf_counter() - t))
        # -----------------------
        # CALC WEIGHT FACTORS
        # -----------------------

        npxl = np.int(all_xp.size / self.timespace.par.snpt ** 2)
        npt = 16
        all_xp = all_xp.reshape((npxl, npt))

        weights = np.zeros((npxl, npt), dtype=int)
        indices = np.zeros((npxl, npt), dtype=int)
        print(f'all_xp shape: {all_xp.shape}')
        t = tm.perf_counter()

        self.compress_vector_njit(all_xp, npxl, npt, weights, indices)

        print('Calc wf: ' + str(tm.perf_counter() - t))

        # -----------------------
        # CALC REVERSED WEIGHT FACTORS
        # -----------------------
        # t = tm.perf_counter()
        # indices, weights = self.calc_rmw(all_xp)
        # print('Calc rwf: ' + str(tm.perf_counter() - t))
        # del all_xp
        # -----------------------


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
                # compare_mi = mapsi[np.int(actmaps / 2): actmaps]
                # compare_mw = mapsweight[np.int(actmaps / 2): actmaps]
                # del mapsi
                # del mapsweight
                # indicies, weights = self._new_calc_wf(xp)  # New wf func.
                # diff = indicies - compare_mi
                # print('mapsi')
                # print(np.sum(diff))
                # diff = weights - compare_mw
                # print('mapsw')
                # print(np.sum(diff))
                # raise SystemExit
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

    # ============================================================
    # NEW FUNCTIONS
    # ============================================================

    # Longtrack
    # -------------
    #@njit
    def longtrack_all(self, initial_xp, initial_yp, film):
        tpar = self.timespace.par

        large_xp = np.zeros((tpar.profile_count, len(initial_xp)))
        large_xp[film] = initial_xp

        direction = 1
        endprofile = tpar.profile_count
        turn_now = film * tpar.dturns
        for twice in range(2):
            yp = initial_yp
            for profile in range(film + direction, endprofile, direction):
                print(f'tracking from {profile} to {profile + 1}')

                denergy = (yp - tpar.yat0) * self.mapinfo.dEbin
                if direction > 0:
                    dphi = ((large_xp[profile - 1] + tpar.x_origin) * tpar.h_num
                            * tpar.omega_rev0[turn_now] * tpar.dtbin
                            - tpar.phi0[turn_now])
                    for i in range(tpar.dturns):
                        dphi -= tpar.dphase[turn_now] * denergy
                        turn_now += 1
                        denergy += calc_denergy(tpar.q, tpar.vrf1,
                                                tpar.vrf1dot, tpar.vrf2,
                                                tpar.vrf2dot,
                                                tpar.time_at_turn,
                                                turn_now, dphi,
                                                tpar.phi0,
                                                tpar.h_ratio,
                                                tpar.phi12,
                                                tpar.deltaE0)
                else:
                    dphi = ((large_xp[profile + 1] + tpar.x_origin) * tpar.h_num
                            * tpar.omega_rev0[turn_now] * tpar.dtbin
                            - tpar.phi0[turn_now])
                    for i in range(tpar.dturns):
                        denergy -= calc_denergy(tpar.q, tpar.vrf1,
                                                tpar.vrf1dot, tpar.vrf2,
                                                tpar.vrf2dot,
                                                tpar.time_at_turn,
                                                turn_now, dphi,
                                                tpar.phi0,
                                                tpar.h_ratio,
                                                tpar.phi12,
                                                tpar.deltaE0)
                        turn_now -= 1
                        dphi += tpar.dphase[turn_now] * denergy

                large_xp[profile] = ((dphi + tpar.phi0[turn_now])
                                     / (float(tpar.h_num)
                                        * tpar.omega_rev0[turn_now]
                                        * tpar.dtbin)
                                     - tpar.x_origin)
                yp = denergy / float(self.mapinfo.dEbin) + tpar.yat0

            direction = -1
            endprofile = -1

        return large_xp

    def longtrack_one_particle(self, init_points):

        # TEMP
        rec_prof = 0
        # END TEMP

        tpar = self.timespace.par
        all_turns = tpar.dturns * (tpar.profile_count - 1)

        out_xp = np.zeros(tpar.profile_count)
        out_xp[0] = init_points[0]

        # If things are going to be run in parallel, can i move this out of the function, and
        # make a (2?)D array (initial?) with dphis for all points?
        dphi = ((out_xp[0] + tpar.x_origin)
                * tpar.h_num
                * tpar.omega_rev0[rec_prof]
                * tpar.dtbin
                - tpar.phi0[rec_prof])
        denergy = (init_points[1] - tpar.yat0) * self.mapinfo.dEbin

        # This test version will always start from 1
        xp_idx = 1
        turn = rec_prof
        while turn < all_turns:
            dphi -= tpar.dphase[turn] * denergy
            turn += 1
            denergy += calc_denergy(tpar.q, tpar.vrf1,
                                    tpar.vrf1dot, tpar.vrf2,
                                    tpar.vrf2dot,
                                    tpar.time_at_turn,
                                    turn, dphi,
                                    tpar.phi0,
                                    tpar.h_ratio,
                                    tpar.phi12,
                                    tpar.deltaE0)
            if turn % tpar.dturns == 0:
                # Calculating xp at profile_measurement
                out_xp[xp_idx] = ((dphi + tpar.phi0[turn])
                                  / (float(tpar.h_num)
                                     * tpar.omega_rev0[turn]
                                     * tpar.dtbin)
                                  - tpar.x_origin)
                # Calculating yp
                yp = denergy / float(self.mapinfo.dEbin) + tpar.yat0

                # Calculating start dphi and denergi for this profile
                dphi = ((out_xp[xp_idx] + tpar.x_origin)
                        * tpar.h_num
                        * tpar.omega_rev0[turn]
                        * tpar.dtbin
                        - tpar.phi0[turn])
                denergy = (yp - tpar.yat0) * self.mapinfo.dEbin
                xp_idx += 1
        return out_xp

    def longtrack_one_particle_mod(self, args):

        # TEMP
        rec_prof = 0
        # END TEMP
        tpar = self.timespace.par

        out_xp = np.zeros(tpar.profile_count)
        # out_xp[0] = init_points[0]

        # If things are going to be run in parallel, can i move this out of the function, and
        # make a (2?)D array (initial?) with dphis for all points?
        # dphi = ((out_xp[0] + tpar.x_origin)
        #         * tpar.h_num
        #         * tpar.omega_rev0[rec_prof]
        #         * tpar.dtbin
        #         - tpar.phi0[rec_prof])
        # denergy = (init_points[1] - tpar.yat0) * self.mapinfo.dEbin

        dphi = args[0]
        denergy = args[1]

        # This test version will always start from 1
        self.track_positive(rec_prof, out_xp, self.all_turns, tpar.dphase,
                            denergy, dphi, tpar.q, tpar.vrf1,
                            tpar.vrf1dot, tpar.vrf2, tpar.vrf2dot,
                            tpar.time_at_turn, tpar.phi0, tpar.h_ratio,
                            tpar.phi12, tpar.deltaE0, tpar.dturns,
                            tpar.omega_rev0, tpar.dtbin, tpar.x_origin,
                            self.mapinfo.dEbin, tpar.yat0, tpar.h_num)
        return out_xp

    @staticmethod
    @njit(fastmath=True)
    def track_positive(rec_prof, out_xp, all_turns, dphase, denergy,
                       dphi, q, vrf1, vrf1dot, vrf2, vrf2dot,
                       time_at_turn, phi0, h_ratio, phi12, deltaE0, dturns,
                       omega_rev0, dtbin, x_origin, dEbin, yat0, h_num):
        xp_idx = 1
        turn = rec_prof
        while turn < all_turns:
            dphi -= dphase[turn] * denergy
            turn += 1
            denergy += calc_denergy(q, vrf1,
                                    vrf1dot, vrf2,
                                    vrf2dot,
                                    time_at_turn,
                                    turn, dphi,
                                    phi0,
                                    h_ratio,
                                    phi12,
                                    deltaE0)
            if turn % dturns == 0:
                # Calculating xp at profile_measurement
                out_xp[xp_idx] = ((dphi + phi0[turn])
                                  / (float(h_num)
                                     * omega_rev0[turn]
                                     * dtbin)
                                  - x_origin)
                # Calculating yp
                yp = denergy / float(dEbin) + yat0

                # Calculating start dphi and denergi for this profile
                dphi = ((out_xp[xp_idx] + x_origin)
                        * h_num
                        * omega_rev0[turn]
                        * dtbin
                        - phi0[turn])
                denergy = (yp - yat0) * dEbin
                xp_idx += 1

    def calc_dphi_denergy(self, xp, yp, turn):
        tpar = self.timespace.par
        dphi = ((xp + tpar.x_origin)
                * tpar.h_num
                * tpar.omega_rev0[turn]
                * tpar.dtbin
                - tpar.phi0[turn])
        denergy = (yp - tpar.yat0) * self.mapinfo.dEbin
        return dphi, denergy
    # ----------------- END LONGTRACK ---------------------------------

    # Weight factors
    # -------------
    # Calculating using np.unique
    def wf_alg1(self, inn_xp):
        particles_pr_pxl = self.timespace.par.snpt**2
        nr_pixels = np.int(inn_xp.size / self.timespace.par.snpt**2)

        xp = np.ceil(inn_xp.copy()).astype(int)
        xp = xp.reshape((nr_pixels, particles_pr_pxl))
        xp = xp - 1  # Fortran compensation

        indices = -np.ones((nr_pixels, particles_pr_pxl))
        weights = np.zeros((nr_pixels, particles_pr_pxl))

        i = 0

        # Finn ut om det er mulig aa sette inn en liten array i en stor array!

        for pixel in xp:
            unique, counts = np.unique(pixel, return_counts=True)
            indices[i, :len(unique)] = unique
            weights[i, :len(counts)] = counts
            i += 1

        # i = 0
        # for pixel in xp:
        #     for coordinate in pixel:
        #         bins[i, coordinate] += 1
        #     i += 1

        # indices = -np.ones((nr_pixels, particles_pr_pxl))
        # weights = np.zeros((nr_pixels, particles_pr_pxl))

        # i = 0
        # for bin in bins:
        #     found_idc = np.nonzero(bin)[0]
        #     insert_idc = np.where(found_idc >= 0)[0]
        #     indices[i, insert_idc] = found_idc
        #     weights[i, insert_idc] = bin[found_idc]
        #     i += 1

        # return indices, weights

        # k = 0
        # for bin in bins:
        #     j = 0
        #     for i in range(len(bin)):
        #         if bin[i] != 0:
        #             indices[k, j] = i - 1  # to compensate for fortran
        #             weights[k, j] = bin[i]
        #             j += 1
        #             if not any(bin[i + 1:]):
        #                 k += 1
        #                 break
        # return indices, weights

    # New calculate wf algorithm for MULTIPLE vectors
    def wf_alg2(self, all_xp):
        particles_pr_pxl = self.timespace.par.snpt ** 2
        nr_pxl = np.int(np.ceil(all_xp.size / self.timespace.par.snpt ** 2))

        xp = np.ceil(all_xp).astype(int)
        xp = xp.reshape((nr_pxl, particles_pr_pxl))
        xp = xp - 1  # Fortran compensation

        bins = np.zeros((nr_pxl, np.max(xp) + 1))

        # Hmhmhmh
        # ----------------------------
        i = 0
        for pixel in xp:
            for coordinate in pixel:
                bins[i, coordinate] += 1
            i += 1
        # ----------------------------

        indices = -np.ones((all_xp.size, particles_pr_pxl))
        weights = np.zeros((all_xp.size, particles_pr_pxl))

        i = 0
        for bin in bins:
            found_idc = np.nonzero(bin)[0]
            insert_idc = np.where(found_idc >= 0)[0]
            indices[i, insert_idc] = found_idc
            weights[i, insert_idc] = bin[found_idc]
            i += 1

    @staticmethod
    # @njit(fastmath=True, parallel=True)
    # using
    def wf_alg3(xp, particles_pr_pxl):
        nr_pxl = np.int(np.ceil(xp.size / particles_pr_pxl))
        max_coordinate = np.max(xp) + 1
        pxl_array = xp.reshape((nr_pxl, particles_pr_pxl))
        pxl_array = pxl_array - 1  # Fortran compensation

        bins = np.zeros((nr_pxl, max_coordinate))

        t0 = tm.perf_counter()
        Reconstruct.find_accumulate_pixelvals(pxl_array, nr_pxl,
                                              particles_pr_pxl, bins)
        print('binning arrays: ' + str(tm.perf_counter() - t0))

        indices = -np.ones((pxl_array.size, particles_pr_pxl))
        weights = np.zeros((pxl_array.size, particles_pr_pxl))

        dt1 = 0
        dt2 = 0
        dt3 = 0
        # t0 = tm.perf_counter()
        for pxl_idx in range(nr_pxl):
            t = tm.perf_counter()
            not_zero = np.where(bins[pxl_idx] != 0)[0]
            dt1 += (tm.perf_counter() - t)

            t = tm.perf_counter()
            indices[pxl_idx, 0:len(not_zero)] = not_zero
            dt2 += (tm.perf_counter() - t)

            t = tm.perf_counter()
            weights[pxl_idx, 0:len(not_zero)] = bins[pxl_idx, not_zero]
            dt3 += (tm.perf_counter() - t)
        # print('filling arrays: ' + str(tm.perf_counter() - t0))

        print(f'dt1: {str(dt1/pxl_idx)}')
        print(f'dt2: {str(dt2/pxl_idx)}')
        print(f'dt3: {str(dt3/pxl_idx)}')
        print(f'pxls: {str(pxl_idx)}')

        # print(indices[0])
        # print(weights[0])

    @staticmethod
    @njit(fastmath=True, parallel=True)
    # Needed for wf_alg3
    def find_accumulate_pixelvals(pxls, nr_pxl, particles_pr_pxl, bins):
        for pixel in range(nr_pxl):
            one_pxl = pxls[pixel]
            for i in range(particles_pr_pxl):
                bins[pixel, one_pxl[i]] += 1

    @staticmethod
    @njit(fastmath=True)
    def compress_vector_njit(pixels, nr_pxl, npt, weights, indices):
        for i in range(nr_pxl):
            already_checked = [0]
            counter = 0
            for j in range(npt):
                if pixels[i, j] not in already_checked:
                    already_checked.append(pixels[i, j])
                    weights[i, counter] = count(pixels[i], pixels[i, j])
                    indices[i, counter] = pixels[i, j]
                    counter += 1
        return indices, weights

    # Reversed Weight
    # -------------
    def calc_rmw(self, all_coords):
        all_coords -= 1
        indices = -np.ones((self.timespace.par.profile_count,
                            self.timespace.par.profile_length + 1), dtype=int)
        weights = np.zeros((self.timespace.par.profile_count,
                            self.timespace.par.profile_length + 1), dtype=int)

        i = 0
        for profile_coords in all_coords:
            uniques, counts = np.unique(profile_coords, return_counts=True)
            indices[i, :len(uniques)] = uniques
            weights[i, :len(uniques)] = counts
            i += 1

        return indices, (weights / self.timespace.par.snpt**2)\
                         * self.timespace.par.profile_count

    # ============================================================
    # END NEW FUNCTIONS
    # ============================================================

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

# Needed for 'compress_vector_njit()'
@njit(fastmath=True)
def count(vector, nr):
    count = 0
    for i in prange(len(vector)):
        if vector[i] == nr:
            count += 1
    return count


@njit(fastmath=True, parallel=True)
def calc_denergy(q, vrf1, vrf1dot, vrf2, vrf2dot, time_at_turn,
                 turn_now, dphi, phi0, h_ratio, phi12, deltaE0):
    return q * (vrft(vrf1, vrf1dot, time_at_turn[turn_now])
                * np.sin(dphi + phi0[turn_now])
                + vrft(vrf2, vrf2dot, time_at_turn[turn_now])
                * np.sin(h_ratio * (dphi + phi0[turn_now] - phi12))
                ) - deltaE0[turn_now]
