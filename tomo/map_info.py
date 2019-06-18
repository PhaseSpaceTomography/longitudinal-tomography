import numpy as np
import logging
import physics
from utils.assertions import TomoAssertions as ta
from utils.exceptions import *

class MapInfo:

    def __init__(self, timespace):

        (self.jmin,         # minimum energy
         self.jmax,         # maximum energy
         self.imin,         # Minimum phase in profile
         self.imax,         # Maximum phase in profile
         self.dEbin,        # Phase space pixel height
         self.allbin_min,   # imin and imax as calculated from jmax and jmin
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
                                timespace.par.c1,
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

    # This procedure calculates and sets the limits
    # 	in i (phase) and j (energy).
    # The i-j coordinate system is the one used locally
    # 	(reconstructed phase space).
    # This is the main function of the class
    def _find_ijlimits(self, beam_ref_frame, dturns, profile_length,
                       full_pp_flag, x_origin, dtbin, h_num, omega_rev0,
                       vrf1, vrf1dot, vrf2, vrf2dot, yat0, c1, q, e0,
                       phi0, eta0, demax, beta0, h_ratio, phi12,
                       time_at_turn, filmstart, filmstop, filmstep,
                       profile_mini, profile_maxi):

        turn_now = (beam_ref_frame - 1)*dturns
        indarr = np.arange(profile_length + 1)

        phases = self.calculate_phases_turn(x_origin,
                                            dtbin,
                                            h_num,
                                            omega_rev0[turn_now],
                                            profile_length,
                                            indarr)

        # Calculate  energy pixel size [MeV]
        dEbin = self._energy_binning(vrf1,
                                     vrf1dot,
                                     vrf2,
                                     vrf2dot,
                                     yat0,
                                     c1,
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

        ta.assert_greater(dEbin, 'dEbin', 0, EnergyBinningError)

        if full_pp_flag == 1:
            # Calculates limits for tracking all pixels
            (jmin,
             jmax,
             allbin_min,
             allbin_max) = self._limits_track_allpxl(filmstop,
                                                     profile_length,
                                                     yat0)
        else:
            # Finding limits for tracking all active pixels
            (jmin,
             jmax,
             allbin_min,
             allbin_max) = self._extrema_active_pxlenergy(filmstart,
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
                                                          c1,
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

    # Calculates, and returns dEbin
    def _energy_binning(self, vrf1, vrf1dot, vrf2, vrf2dot, yat0, c1,
                        profile_length, q, e0, phi0, h_num, eta0,
                        dtbin, omega_rev0, demax, beta0, h_ratio,
                        phi12, time_at_turn, phases, turn):
        delta_e_known = 0.0
        if demax < 0.0:
            if physics.vrft(vrf2, vrf2dot, turn) != 0.0:
                # finding maximum energy starting from lowest phase
                # 	and uppermost phase respectively
                energies_low = self.trajectoryheight(phases,
                                                     phases[0],
                                                     delta_e_known,
                                                     q,
                                                     c1,
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
                                                    c1,
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

    # Extrema of active pixels in energy (j-index or y-index) direction.
    def _extrema_active_pxlenergy(self, filmstart, filmstop, filmstep,
                                  dturns, profile_length, indarr, dEbin,
                                  x_origin, dtbin, omega_rev0, h_num, yat0,
                                  q, c1, phi0, vrf1, vrf1dot, vrf2, vrf2dot,
                                  h_ratio, phi12, time_at_turn):
        # index of min and max energy (in bins)
        allbin_min = np.zeros(filmstop, dtype=int)
        allbin_max = np.zeros(filmstop, dtype=int)

        # initializing matrix with zeroes.
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
                                            c1,
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

    # Function for finding jmax for each bin in profile
    # Variable inputs for each profile are turn and phases
    def _find_jmax(self, profile_length, yat0, q, c1, phi0,
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
                                            c1,
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
                                            c1,
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

    # Function for finding jmin for each bin in profile
    # Variable input for each profile is array of jmax
    def _find_jmin(self, yat0, jmax_array, threshold=1):
        jmin_array = np.ceil(2.0 * yat0 - jmax_array[:] - 0.5)
        #  Check each element if less than threshold,
        #  	if true, threshold will be used.
        return np.where(jmin_array[:] >= threshold, jmin_array[:], threshold)

    # Finding minimum and maximum of all bin
    def _find_allbin_min(self, jmin_array, jmax_array, profile_length):
        for i in range(0, profile_length):
            if jmax_array[i] - jmin_array[i] >= 0:
                return i

    def _find_allbin_max(self, jmin_array, jmax_array, profile_length):
        for i in range(profile_length - 1, 0, -1):
            if jmax_array[i] - jmin_array[i] >= 0:
                return i

    # Set limits in i and j if full_PP_track flag is high\
    # Will track all pixels.
    def _limits_track_allpxl(self, filmstop, profile_length, yat0):

        # initializing matrix with zeroes.
        allbin_min = np.zeros(filmstop, dtype=int)
        allbin_max = np.zeros(filmstop, dtype=int)
        jmax = np.zeros((filmstop, profile_length), dtype=np.int32)
        jmin = np.copy(jmax)

        jmax[:, :] = profile_length
        jmin[:, :] = np.ceil(2.0 * yat0 - jmax[:] + 0.5)

        allbin_min[0] = int(0)
        allbin_max[0] = int(profile_length)
        return jmin, jmax, allbin_min, allbin_max

    # Adjustment of limits in relation to
    # 	specified input min/max index and found max/min in profile.
    # 	E.g. if profileMinIndex is greater than allbin_min, use profileMinIndex.
    # Calculates limits in i axis.
    def _adjust_limits(self, filmstart, filmstop, filmstep,
                       full_pp_flag, profile_mini, profile_maxi,
                       yat0, profile_length, jmax, jmin,
                       allbin_min, allbin_max):
        # imin, imax: Minimum and maximum values at phase axis for profiles
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

    # Returns an array of phases for the given turn
    def calculate_phases_turn(self, x_origin, dtbin, h_num,
                              omega_rev0_at_turn, profile_length,
                              indarr):
        # phases: phase at the edge of each bin along the i-axis
        phases = np.zeros(profile_length + 1)
        phases[0] = (x_origin
                     * dtbin
                     * h_num
                     * omega_rev0_at_turn)
        phases[0:] = ((x_origin + indarr)
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

    # This is a trajectory height calculator given a phase and energy.
    @staticmethod
    def trajectoryheight(phi, phi_known, delta_e_known, q, c1,
                         phi0, vrf1, vrf1dot, vrf2, vrf2dot,
                         h_ratio, phi12, time_at_turn, turn_now):
        temp1 = delta_e_known**2
        temp2 = 2.0 * q / float(c1[turn_now])
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
