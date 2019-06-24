import numpy as np
import logging
import physics
from utils.assertions import TomoAssertions as ta
from utils.exceptions import (EnergyBinningError,
                              EnergyLimitsError,
                              PhaseLimitsError,
                              MapCreationError)
# ===============
# About the class
# ===============
# There are two reference systems in the tomography routine:
#    i) The reconstructed phase space coordinate system which has its origin at some minimum phase,
#        where the x-bins are dtbin wide and the y-bins are dEbin wide.
#    ii) The physical coordinates expressed in energy energy and phase wrt. the sync. particle
#
#
# This class produces and sets the limits in i (phase) and j (energy).
# The i-j coordinate system is the one used locally for the reconstructed phase space.
#
# =======
# Example
# =======
#
#
# ^ j (energy)
# |xxxxxxxxxxxxxxxxxxxx
# |xxxxxxxxxxxxxxxxxxxx
# |xxxxxxxx...xxxxxxxxx
# |xxxxxx.......xxxxxxx
# |xxxxx.........xxxxxx
# |xxxxxx.......xxxxxxx
# |xxxxxxxx...xxxxxxxxx
# |xxxxxxxxxxxxxxxxxxxx     i (phase)
# -------------------------->
#
# x - not active pixel (unless full_pp_flag)
# o - active pixel
#
# If full_pp_flag is true will all the pixels, marked both wit 'x' and '.' be tracked.
# Else, will only the picels marked with '.' be tracked.
# The tracked area is made by the out of the combination of the i and j limits.
#
# ================
# Object variables
# ================
#
# imin, imax        Lower and upper limit in 'i' (phase) of the i-j coordinate system.
# jmin, jmax        Lower and upper limit in 'j' (energy) of the i-j coordinate system.
# dEbin             Phase space pixel height (in j direction) [MeV]
# allbin_min/max    imin and imax as calculated from jmax and jmin.
#                     Used unless they fall outside the borders of profile_mini or -_maxi stated in parameters,
#                     or if full_pp_flag (track all pixels) is true.


class MapInfo:

    def __init__(self, timespace):

        (self.jmin,
         self.jmax,
         self.imin,
         self.imax,
         self.dEbin,
         self.allbin_min,
         self.allbin_max) = self._find_ijlimits(
                                timespace.par.beam_ref_frame,
                                timespace.par.dturns,
                                timespace.par.profile_length,
                                timespace.par.full_pp_flag,
                                timespace.par.x_origin,
                                timespace.par.dtbin,
                                timespace.par.h_num,
                                timespace.par.omega_rev0,
                                timespace.par.vrf1,
                                timespace.par.vrf1dot,
                                timespace.par.vrf2,
                                timespace.par.vrf2dot,
                                timespace.par.yat0,
                                timespace.par.dphase,
                                timespace.par.q,
                                timespace.par.e0,
                                timespace.par.phi0,
                                timespace.par.eta0,
                                timespace.par.demax,
                                timespace.par.beta0,
                                timespace.par.h_ratio,
                                timespace.par.phi12,
                                timespace.par.time_at_turn,
                                timespace.par.filmstart,
                                timespace.par.filmstop,
                                timespace.par.filmstep,
                                timespace.par.profile_mini,
                                timespace.par.profile_maxi)
        self._assert_correct_arrays(timespace)

    # Main function for the class. finds limits in i and j axis.
    # Local variables:
    #   - turn_now: machine turn (=0 at profile = 0)
    #   - indarr: array holding numbers from 0 to (profile length + 1)
    #   - phases: phase at the edge of each bin along the i-axis
    def _find_ijlimits(self, beam_ref_frame, dturns, profile_length,
                       full_pp_flag, x_origin, dtbin, h_num, omega_rev0,
                       vrf1, vrf1dot, vrf2, vrf2dot, yat0, dphase, q, e0,
                       phi0, eta0, demax, beta0, h_ratio, phi12,
                       time_at_turn, filmstart, filmstop, filmstep,
                       profile_mini, profile_maxi):

        turn_now = (beam_ref_frame - 1) * dturns
        indarr = np.arange(profile_length + 1)

        phases = self.calculate_phases_turn(x_origin,
                                            dtbin,
                                            h_num,
                                            omega_rev0[turn_now],
                                            profile_length,
                                            indarr)

        dEbin = self._energy_binning(vrf1,
                                     vrf1dot,
                                     vrf2,
                                     vrf2dot,
                                     yat0,
                                     dphase,
                                     profile_length,
                                     q,
                                     e0,
                                     phi0,
                                     h_num,
                                     eta0,
                                     dtbin,
                                     omega_rev0,
                                     demax,
                                     beta0,
                                     h_ratio,
                                     phi12,
                                     time_at_turn,
                                     phases,
                                     turn_now)

        ta.assert_greater(dEbin, 'dEbin', 0.0, EnergyBinningError)

        if full_pp_flag == 1:
            (jmin,
             jmax,
             allbin_min,
             allbin_max) = self._limits_track_all_pxl(filmstop,
                                                      profile_length,
                                                      yat0)
        else:
            (jmin,
             jmax,
             allbin_min,
             allbin_max) = self._limits_track_active_pxl(filmstart,
                                                         filmstop,
                                                         filmstep,
                                                         dturns,
                                                         profile_length,
                                                         indarr,
                                                         dEbin,
                                                         x_origin,
                                                         dtbin,
                                                         omega_rev0,
                                                         h_num,
                                                         yat0,
                                                         q,
                                                         dphase,
                                                         phi0,
                                                         vrf1,
                                                         vrf1dot,
                                                         vrf2,
                                                         vrf2dot,
                                                         h_ratio,
                                                         phi12,
                                                         time_at_turn)

        # Calculate limits (index of bins) in i-axis (phase axis),
        # 	adjust j-axis (energy axis)
        (jmin,
         jmax,
         imin,
         imax) = self._adjust_limits(filmstart,
                                     filmstop,
                                     filmstep,
                                     full_pp_flag,
                                     profile_mini,
                                     profile_maxi,
                                     yat0,
                                     profile_length,
                                     jmax,
                                     jmin,
                                     allbin_min,
                                     allbin_max)

        return jmin, jmax, imin, imax, dEbin, allbin_min, allbin_max

    # Calculating the difference of energy of one pixel.
    # This will be the height of each pixel in the physical coordinate system
    def _energy_binning(self, vrf1, vrf1dot, vrf2, vrf2dot, yat0, dphase,
                        profile_length, q, e0, phi0, h_num, eta0,
                        dtbin, omega_rev0, demax, beta0, h_ratio,
                        phi12, time_at_turn, phases, turn):
        delta_e_known = 0.0
        ta.assert_not_equal(demax, 'dEmax',
                            0.0, EnergyBinningError,
                            'The specified maximum energy of '
                            'reconstructed phase space is invalid.')
        if demax < 0.0:
            if physics.vrft(vrf2, vrf2dot, turn) != 0.0:
                energies_low = self.trajectoryheight(phases,
                                                     phases[0],
                                                     delta_e_known,
                                                     q,
                                                     dphase,
                                                     phi0,
                                                     vrf1,
                                                     vrf1dot,
                                                     vrf2,
                                                     vrf2dot,
                                                     h_ratio,
                                                     phi12,
                                                     time_at_turn,
                                                     turn)

                energies_up = self.trajectoryheight(phases,
                                                    phases[profile_length],
                                                    delta_e_known,
                                                    q,
                                                    dphase,
                                                    phi0,
                                                    vrf1,
                                                    vrf1dot,
                                                    vrf2,
                                                    vrf2dot,
                                                    h_ratio,
                                                    phi12,
                                                    time_at_turn,
                                                    turn)

                return (min(np.amax(energies_low), np.amax(energies_up))
                        / (profile_length - yat0))
            else:
                return (beta0[turn]
                        * np.sqrt(e0[turn]
                                  * q
                                  * physics.vrft(vrf1, vrf1dot, turn)
                                  * np.cos(phi0[turn])
                                  / (2 * np.pi * h_num * eta0[turn]))
                        * dtbin
                        * h_num
                        * omega_rev0[turn])
        else:
            return float(demax) / (profile_length - yat0)

    # Finding limits for tracking all pixels in reconstructed phase space.
    def _limits_track_all_pxl(self, filmstop, profile_length, yat0):
        allbin_min = np.zeros(filmstop, dtype=np.int32)
        allbin_max = np.zeros(filmstop, dtype=np.int32)

        jmax = np.zeros((filmstop, profile_length), dtype=np.int32)
        jmin = np.copy(jmax)

        jmax[:, :] = profile_length
        jmin[:, :] = np.ceil(2.0 * yat0 - jmax[:] + 0.5)

        allbin_min[0] = np.int32(0)
        allbin_max[0] = np.int32(profile_length)
        return jmin, jmax, allbin_min, allbin_max

    # Finding limits for tracking active pixels (stated in parameters)
    def _limits_track_active_pxl(self, filmstart, filmstop, filmstep,
                                  dturns, profile_length, indarr, dEbin,
                                  x_origin, dtbin, omega_rev0, h_num, yat0,
                                  q, dphase, phi0, vrf1, vrf1dot, vrf2, vrf2dot,
                                  h_ratio, phi12, time_at_turn):
        allbin_min = np.zeros(filmstop)
        allbin_max = np.zeros(filmstop)

        jmax = np.zeros((filmstop, profile_length), dtype=np.int32)
        jmin = np.copy(jmax)

        for film in range(filmstart - 1, filmstop, filmstep):
            turn = film * dturns

            phases = self.calculate_phases_turn(x_origin,
                                                dtbin,
                                                h_num,
                                                omega_rev0[turn],
                                                profile_length,
                                                indarr)

            jmax[film, :] = self._find_jmax(profile_length,
                                            yat0,
                                            q,
                                            dphase,
                                            phi0,
                                            vrf1,
                                            vrf1dot,
                                            vrf2,
                                            vrf2dot,
                                            h_ratio,
                                            phi12,
                                            time_at_turn,
                                            phases,
                                            turn,
                                            dEbin)

            jmin[film, :] = self._find_jmin(yat0, jmax[film, :])

            allbin_min[film] = self._find_allbin_min(
                                        jmin[film, :],
                                        jmax[film, :],
                                        profile_length)
            allbin_max[film] = self._find_allbin_max(
                                        jmin[film, :],
                                        jmax[film, :],
                                        profile_length)

        return jmin, jmax, allbin_min, allbin_max

    # Function for finding maximum energy (j max) for each bin in the profile
    def _find_jmax(self, profile_length, yat0, q, dphase, phi0,
                   vrf1, vrf1dot, vrf2, vrf2dot, h_ratio,
                   phi12, time_at_turn, phases, turn, dEbin):

        energy = 0.0
        jmax_low = np.zeros(profile_length + 1)
        jmax_up = np.zeros(profile_length + 1)

        # finding max energy at edges of profiles
        for i in range(profile_length + 1):
            temp_energy = np.floor(yat0
                                   + self.trajectoryheight(
                                            phases[i],
                                            phases[0],
                                            energy,
                                            q,
                                            dphase,
                                            phi0,
                                            vrf1,
                                            vrf1dot,
                                            vrf2,
                                            vrf2dot,
                                            h_ratio,
                                            phi12,
                                            time_at_turn,
                                            turn) / dEbin)
            jmax_low[i] = int(temp_energy)

            temp_energy = np.floor(yat0
                                   + self.trajectoryheight(
                                            phases[i],
                                            phases[profile_length],
                                            energy,
                                            q,
                                            dphase,
                                            phi0,
                                            vrf1,
                                            vrf1dot,
                                            vrf2,
                                            vrf2dot,
                                            h_ratio,
                                            phi12,
                                            time_at_turn,
                                            turn) / dEbin)
            jmax_up[i] = int(temp_energy)

        jmax = np.zeros(profile_length)
        for i in range(profile_length):
            jmax[i] = min([jmax_up[i], jmax_up[i + 1],
                           jmax_low[i], jmax_low[i + 1],
                           profile_length])
        return jmax

    # Function for finding minimum energy (j min) for each bin in profile
    # Checking each element if less than threshold, in such cases will threshold be used.
    def _find_jmin(self, yat0, jmax_array, threshold=1):
        jmin_array = np.ceil(2.0 * yat0 - jmax_array[:] - 0.5)
        return np.where(jmin_array[:] >= threshold, jmin_array[:], threshold)

    # Finding index for minimum phase for profile
    def _find_allbin_min(self, jmin_array, jmax_array, profile_length):
        for i in range(0, profile_length):
            if jmax_array[i] - jmin_array[i] >= 0:
                return i

    # Finding index for maximum phase for profile
    def _find_allbin_max(self, jmin_array, jmax_array, profile_length):
        for i in range(profile_length - 1, 0, -1):
            if jmax_array[i] - jmin_array[i] >= 0:
                return i

    # Adjustment of limits in relation to
    # 	specified input min/max index and found max/min in profile.
    # 	E.g. if profile_mini is greater than allbin_min, use profile_mini.
    # Calculates limits in i axis.
    def _adjust_limits(self, filmstart, filmstop, filmstep,
                       full_pp_flag, profile_mini, profile_maxi,
                       yat0, profile_length, jmax, jmin,
                       allbin_min, allbin_max):
        imin = np.zeros(filmstop, dtype=int)
        imax = np.zeros(filmstop, dtype=int)
        for film in range(filmstart - 1, filmstop, filmstep):
            if profile_mini > allbin_min[film] or full_pp_flag:
                imin[film] = profile_mini
                jmax[film, 0:profile_mini] = np.floor(yat0)
                jmin[film, :] = np.ceil(2.0 * yat0 - jmax[film, :] + 0.5)
            else:
                imin[film] = allbin_min[film]

            if profile_maxi < allbin_max[film] or full_pp_flag:
                imax[film] = profile_maxi
                jmax[film, profile_maxi: profile_length] = np.floor(yat0)
                jmin[film, :] = np.ceil(2.0 * yat0 - jmax[film, :] + 0.5)
            else:
                imax[film] = allbin_max[film]

        return jmin, jmax, imin, imax

    # Returns an array of phases for a given turn
    def calculate_phases_turn(self, x_origin, dtbin, h_num,
                              omega_rev0_at_turn, profile_length,
                              indarr):
        ta.assert_equal(len(indarr),
                        'index array length',
                        profile_length + 1,
                        MapCreationError,
                        'The index array should have length '
                        'profile_length + 1')
        phases = ((x_origin + indarr)
                  * dtbin
                  * h_num
                  * omega_rev0_at_turn)
        return phases

    def _assert_correct_arrays(self, ts):
        for film in range(ts.par.filmstart - 1,
                          ts.par.filmstop,
                          ts.par.filmstep):

            # Testing imin and imax
            ta.assert_inrange(self.imin[film], 'imin', 0, self.imax[film],
                              PhaseLimitsError,
                              f'imin and imax out of bounds '
                              f'at film: {film}')
            ta.assert_less_or_equal(self.imax[film], 'imax',
                                    self.jmax[film].size,
                                    PhaseLimitsError,
                                    f'imin and imax out of bounds '
                                    f'at film: {film}')

            # Testing jmin and jmax
            ta.assert_array_in_range(self.jmin[film,
                                               self.imin[film]:
                                               self.imax[film]], 0,
                                     self.jmax[film,
                                               self.imin[film]:
                                               self.imax[film]],
                                     EnergyLimitsError,
                                     msg=f'jmin and jmax out of bounds '
                                         f'at film: {film}',
                                     index_offset=self.imin[film])
            ta.assert_array_less_eq(self.jmax[film,
                                              self.imin[film]:
                                              self.imax[film]],
                                    ts.par.profile_length,
                                    EnergyLimitsError,
                                    f'jmin and jmax out of bounds '
                                    f'at film: {film}')

    # Trajectory height calculator
    @staticmethod
    def trajectoryheight(phi, phi_known, delta_e_known, q, dphase,
                         phi0, vrf1, vrf1dot, vrf2, vrf2dot,
                         h_ratio, phi12, time_at_turn, turn_now):
        temp1 = delta_e_known**2
        temp2 = 2.0 * q / float(dphase[turn_now])
        temp3 = (physics.vrft(vrf1, vrf1dot, turn_now)
                 * (np.cos(phi) - np.cos(phi_known))
                 + physics.vrft(vrf2, vrf2dot, turn_now)
                 * (np.cos(h_ratio * (phi - phi12))
                    - np.cos(h_ratio
                             * (phi_known - phi12)))
                 / h_ratio
                 + (phi - phi_known)
                 * physics.short_rf_voltage_formula(
                            phi0[turn_now], vrf1, vrf1dot, vrf2, vrf2dot,
                            h_ratio, phi12, time_at_turn, turn_now))

        ans = temp1 + temp2 * temp3

        if np.size(ans) > 1:
            # Returning array
            ans = np.array(ans, dtype=complex)
            complex_height = np.sqrt(ans)
        else:
            # Returning scalar
            complex_height = np.sqrt(complex(ans))

        return complex_height.real

    @classmethod
    def write_jmax_tofile(cls, time_space, mapinfo, dir):
        full_path = dir + "py_jmax.dat"
        with open(full_path, "w") as outFile:
            for profile in range(time_space.par.filmstart,
                                 time_space.par.filmstop + 1,
                                 time_space.par.filmstep):
                for i in range(0, time_space.par.profile_length):
                    outFile.write(str(i) + "\t"
                                  + str(mapinfo.jmax[profile - 1, i]) + "\n")
        logging.info("Written jmax to: " + full_path)

    @classmethod
    def write_plotinfo_tofile(cls, time_space, mapinfo, dir):
        # Reconstructing the same output as the FORTRAN code.
        # Writes data needed for plots on a file.
        full_path = dir + "py_plotinfo.dat"
        with open(full_path, "w") as outFile:
            outFile.write("Number of profiles used in each reconstruction,\n")
            outFile.write("profile_count = " +
                          str(time_space.par.profile_count) + "\n")
            outFile.write("Width (in pixels) of each image "
                          + " = length (in bins) of each profile,\n")
            outFile.write("profile_length = "
                          + str(time_space.par.profile_length) + "\n")
            outFile.write("Width (in s) of each pixel "
                          + "= width of each profile bin,\n")
            outFile.write("dtbin = " + str(time_space.par.dtbin) + "\n")
            outFile.write("Height (in eV) of each pixel,\n")
            outFile.write("dEbin = " + str(mapinfo.dEbin) + "\n")
            outFile.write("Number of elementary charges in each image,\n")
            outFile.write("Beam reference profile charge = "
                          + str(time_space.profile_charge) + "\n")
            outFile.write("Position (in pixels) of the "
                          + "reference synchronous point:\n")
            outFile.write("xat0 = " + str(time_space.par.xat0) + "\n")
            outFile.write("yat0 = " + str(time_space.par.yat0) + "\n")
            outFile.write("Foot tangent fit results (in bins):" + "\n")
            outFile.write("tangentfootl = "
                          + str(time_space.par.tangentfoot_low) + "\n")
            outFile.write("tangentfootu = "
                          + str(time_space.par.tangentfoot_up) + "\n")
            outFile.write("fit xat0 = "+ str(time_space.par.fit_xat0) + "\n")
            outFile.write("Synchronous phase (in radians):" + "\n")
            for p in range(time_space.par.filmstart - 1,
                           time_space.par.filmstop, time_space.par.filmstep):
                outFile.write("phi0(" + str(p) + ") = "
                              + str(time_space.par.phi0[(p)* time_space.par.dturns])
                              + "\n")
            outFile.write("Horizontal range (in pixels) "
                          + " of the region in phase space of mapinfo elements:"
                          + "\n")
            for p in range(time_space.par.filmstart - 1,
                           time_space.par.filmstop, time_space.par.filmstep):
                outFile.write("imin(" + str(p) + ") = " + str(mapinfo.imin[p])
                              + ", imax(" + str(p) + ") = " + str(mapinfo.imax[p])
                              + "\n")

        logging.info("Written profile info to: " + full_path)
