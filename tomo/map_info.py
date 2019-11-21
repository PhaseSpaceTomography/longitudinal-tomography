import numpy as np
import sys
import logging
from physics import vrft, short_rf_voltage_formula
from utils.assertions import (assert_greater,
                              assert_not_equal,
                              assert_equal,
                              assert_inrange,
                              assert_less_or_equal,
                              assert_array_in_range,
                              assert_array_less_eq)
from utils.exceptions import (EnergyBinningError,
                              EnergyLimitsError,
                              PhaseLimitsError,
                              MapCreationError,
                              ArrayLengthError)
# ===============
# About the class
# ===============
# There are two reference systems in the tomography routine:
#    i) The reconstructed phase space coordinate system which has its
#        origin at some minimum phase,
#        where the x-bins are dtbin wide and the y-bins are dEbin wide.
#    ii) The physical coordinates expressed in energy energy
#        and phase wrt. the sync. particle
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
#                     Used unless they fall outside the borders of
#                     profile_mini or -_maxi stated in parameters,
#                     or if full_pp_flag (track all pixels) is true.


class MapInfo:

    def __init__(self, machine):
        self.machine = machine

        self.jmin = []
        self.jmax = []
        self.imin = -1
        self.imax = -1
        self.dEbin = -1.0
        self.allbin_min = -1 
        self.allbin_max = -1


    # Main function for the class. finds limits in i and j axis.
    # Local variables:
    #   - turn_now: machine turn (=0 at profile = 0)
    #   - indarr: array holding numbers from 0 to (profile length + 1)
    #   - phases: phase at the edge of each bin along the i-axis
    def find_ijlimits(self):

        self.dEbin = self.find_dEbin()

        # If dEbin is less than 0, an error is raised.  
        assert_greater(self.dEbin, 'dEbin', 0.0, EnergyBinningError)

        # Is this still a valid choise with the new method?
        if self.machine.full_pp_flag == 1:
            (jmin,
             jmax,
             allbin_min,
             allbin_max) = self._limits_track_all_pxl()
        else:
            (jmin,
             jmax,
             allbin_min,
             allbin_max) = self._limits_track_active_pxl(self.dEbin)
        
        self.allbin_min = allbin_min
        self.allbin_max = allbin_max

        # Calculate limits (index of bins) in i-axis (phase axis),
        # 	adjust j-axis (energy axis)
        (jmin, jmax,
         self.imin,
         self.imax) = self._adjust_limits(jmax, jmin, allbin_min, allbin_max)

        self.jmin = jmin.astype(np.int32)
        self.jmax = jmax.astype(np.int32)
        
        # Ensuring that the output is valid
        self._assert_correct_arrays()

    def find_dEbin(self):
        # Calculating turn and phases at the beam reference point
        turn = (self.machine.beam_ref_frame - 1) * self.machine.dturns    
        phases = self._calculate_phases(turn)
        return self._calc_energy_pxl(phases, turn)

    # Calculating the difference of energy of one pixel.
    # This will be the height of each pixel in the physical coordinate system
    def _calc_energy_pxl(self, phases, turn):
        delta_e_known = 0.0
        assert_not_equal(self.machine.demax, 'dEmax',
                         0.0, EnergyBinningError,
                         'The specified maximum energy of '
                         'reconstructed phase space is invalid.')
        if self.machine.demax < 0.0:
            if vrft(self.machine.vrf2, self.machine.vrf2dot, turn) != 0.0:
                energies_low = self._trajectoryheight(
                				phases, phases[0], delta_e_known, turn)

                energies_up = self._trajectoryheight(
                				phases, phases[self.machine.nbins],
                                delta_e_known, turn)

                return (min(np.amax(energies_low), np.amax(energies_up))
                        / (self.machine.nbins - self.machine.yat0))
            else:
                return (self.machine.beta0[turn]
                        * np.sqrt(self.machine.e0[turn]
                                  * self.machine.q
                                  * vrft(self.machine.vrf1,
                                         self.machine.vrf1dot, turn)
                                  * np.cos(self.machine.phi0[turn])
                                  / (2 * np.pi * self.machine.h_num
                                     * self.machine.eta0[turn]))
                        * self.machine.dtbin
                        * self.machine.h_num
                        * self.machine.omega_rev0[turn])
        else:
            return (float(self.machine.demax)
                    / (self.machine.nbins - self.machine.yat0))

    # Finding limits for tracking all pixels in reconstructed phase space.
    def _limits_track_all_pxl(self):
        jmax = np.zeros(self.machine.nbins)
        jmin = np.copy(jmax)

        jmax[:] = self.machine.nbins
        jmin[:] = np.ceil(2.0 * self.machine.yat0 - jmax + 0.5)

        allbin_min = np.int32(0)
        allbin_max = np.int32(self.machine.nbins)
        return jmin, jmax, allbin_min, allbin_max

    # Finding limits for tracking active pixels (stated in parameters)
    def _limits_track_active_pxl(self, dEbin):
        jmax = np.zeros(self.machine.nbins)
        jmin = np.copy(jmax)

        # Calculating turn and phases at the filmstart reference point
        turn = (self.machine.filmstart - 1) * self.machine.dturns
        phases = self._calculate_phases(turn)

        # Jmax to int already here? 
        jmax = self._find_jmax(phases, turn, dEbin)

        jmin = self._find_jmin(jmax)

        allbin_min = self._find_allbin_min(jmin, jmax)

        allbin_max = self._find_allbin_max(jmin, jmax)

        return jmin, jmax, allbin_min, allbin_max

    # Function for finding maximum energy (j max) for each bin in the profile
    def _find_jmax(self, phases, turn, dEbin):

        energy = 0.0
        jmax_low = np.zeros(self.machine.nbins + 1)
        jmax_up = np.zeros(self.machine.nbins + 1)

        # finding max energy at edges of profiles
        for i in range(self.machine.nbins + 1):
            temp_energy = np.floor(self.machine.yat0
                                   + self._trajectoryheight(
                                        phases[i], phases[0], energy, turn)
                                   / dEbin)
            
            jmax_low[i] = int(temp_energy)

            temp_energy = np.floor(self.machine.yat0
                                   + self._trajectoryheight(
                                        phases[i],
                                        phases[self.machine.nbins],
                                        energy, turn)
                                   / dEbin)

            jmax_up[i] = int(temp_energy)

        jmax = np.zeros(self.machine.nbins)
        for i in range(self.machine.nbins):
            jmax[i] = min([jmax_up[i], jmax_up[i + 1],
                           jmax_low[i], jmax_low[i + 1],
                           self.machine.nbins])
        return jmax

    # Function for finding minimum energy (j min) for each bin in profile
    # Checking each element if less than threshold,
    # in such cases will threshold be used.
    def _find_jmin(self, jmax, threshold=1):
        jmin = np.ceil(2.0 * self.machine.yat0 - jmax[:] - 0.5)
        return np.where(jmin[:] >= threshold, jmin[:], threshold)

    # Finding index for minimum phase for profile
    def _find_allbin_min(self, jmin, jmax):
        for i in range(0, self.machine.nbins):
            if jmax[i] - jmin[i] >= 0:
                return i

    # Finding index for maximum phase for profile
    def _find_allbin_max(self, jmin, jmax):
        for i in range(self.machine.nbins - 1, 0, -1):
            if jmax[i] - jmin[i] >= 0:
                return i

    # Adjustment of limits in relation to
    # 	specified input min/max index and found min/max in profile.
    # 	E.g. if profile_mini is greater than allbin_min, use profile_mini.
    # Calculates limits in i axis.
    def _adjust_limits(self, jmax, jmin, allbin_min, allbin_max):
        film = self.machine.filmstart - 1
        if (self.machine.profile_mini > allbin_min
                                        or self.machine.full_pp_flag):
            imin = self.machine.profile_mini
            jmax[:self.machine.profile_mini] = np.floor(self.machine.yat0)
            jmin = np.ceil(2.0 * self.machine.yat0 - jmax + 0.5)
        else:
            imin = allbin_min

        if self.machine.profile_maxi < allbin_max or self.machine.full_pp_flag:
            imax = self.machine.profile_maxi - 1 # -1 in order to count from idx 0
            jmax[self.machine.profile_maxi
                 :self.machine.nbins] = np.floor(self.machine.yat0)
            jmin = np.ceil(2.0 * self.machine.yat0 - jmax + 0.5)
        else:
            imax = allbin_max

        return jmin, jmax, imin, imax

    # Returns an array of phases for a given turn
    def _calculate_phases(self, turn):
        indarr = np.arange(self.machine.nbins + 1)
        assert_equal(len(indarr),
                     'index array length',
                     self.machine.nbins + 1,
                     MapCreationError,
                     'The index array should have length '
                     'nbins + 1')
        phases = ((self.machine.xorigin + indarr)
                  * self.machine.dtbin
                  * self.machine.h_num
                  * self.machine.omega_rev0[turn])
        return phases

    def _assert_correct_arrays(self):
        for film in range(self.machine.filmstart - 1,
                          self.machine.filmstop,
                          self.machine.filmstep):

            # Testing imin and imax
            assert_inrange(self.imin, 'imin', 0, self.imax,
                           PhaseLimitsError,
                           f'imin and imax out of bounds')
            assert_less_or_equal(self.imax, 'imax', self.jmax.size,
                                 PhaseLimitsError,
                                 f'imin and imax out of bounds')

            # Testing jmin and jmax
            assert_array_in_range(self.jmin[self.imin:self.imax], 0,
                                  self.jmax[self.imin:self.imax],
                                  EnergyLimitsError,
                                  msg=f'jmin and jmax out of bounds ',
                                  index_offset=self.imin)
            assert_array_less_eq(self.jmax[self.imin:self.imax],
                                 self.machine.nbins,
                                 EnergyLimitsError,
                                 f'jmin and jmax out of bounds ')
            assert_equal(self.jmin.shape, 'jmin',
                         self.jmax.shape, ArrayLengthError,
                         'jmin and jmax should have the same shape')

    # Trajectory height calculator
    def _trajectoryheight(self, phi, phi_known, delta_e_known, turn):
        cplx_height = 2.0 * self.machine.q / float(self.machine.dphase[turn])
        cplx_height *= (vrft(self.machine.vrf1, self.machine.vrf1dot, turn)
                        * (np.cos(phi) - np.cos(phi_known))
                        + vrft(self.machine.vrf2, self.machine.vrf2dot, turn)
                        * (np.cos(self.machine.h_ratio
                                  * (phi - self.machine.phi12))
                        - np.cos(self.machine.h_ratio
                                 * (phi_known - self.machine.phi12)))
                        / self.machine.h_ratio
                        + (phi - phi_known)
                        * short_rf_voltage_formula(
                            self.machine.phi0[turn], self.machine.vrf1,
                            self.machine.vrf1dot, self.machine.vrf2,
                            self.machine.vrf2dot, self.machine.h_ratio,
                            self.machine.phi12, self.machine.time_at_turn,
                            turn))
        cplx_height += delta_e_known**2

        if np.size(cplx_height) > 1:
            # Returning array
            cplx_height = np.array(cplx_height, dtype=complex)
            cplx_height = np.sqrt(cplx_height)
        else:
            # Returning scalar
            cplx_height = np.sqrt(complex(cplx_height))

        return cplx_height.real

    # Needed for tomoscope in the CCC.
    # Written in the same format as the original fortran version.
    # To be moved to tomoIO
    def print_plotinfo_ccc(self, ts):
        rec_prof = self.machine.filmstart - 1 # '-1' Fortran compensation
        rec_turn = rec_prof * self.machine.dturns
        
        out_s = f' plotinfo.data\n'\
                f'Number of profiles used in each reconstruction,\n'\
                  f' profilecount = {ts.machine.nprofiles}\n'\
                f'Width (in pixels) of each image = '\
                  f'length (in bins) of each profile,\n'\
                f' profilelength = {ts.machine.nbins}\n'\
                f'Width (in s) of each pixel = width of each profile bin,\n'\
                f' dtbin = {ts.machine.dtbin:0.4E}\n'\
                f'Height (in eV) of each pixel,\n'\
                f' dEbin = {self.dEbin:0.4E}\n'\
                f'Number of elementary charges in each image,\n'\
                  f' eperimage = '\
                  f'{ts.profile_charge:0.3E}\n'\
                f'Position (in pixels) of the reference synchronous point:\n'\
                f' xat0 =  {ts.machine.xat0:.3f}\n'\
                f' yat0 =  {ts.machine.yat0:.3f}\n'\
                f'Foot tangent fit results (in bins):\n'\
                f' tangentfootl =    {ts.tangentfoot_low:.3f}\n'\
                f' tangentfootu =    {ts.tangentfoot_up:.3f}\n'\
                f' fit xat0 =   {ts.fitted_xat0:.3f}\n'\
                f'Synchronous phase (in radians):\n'\
                f' phi0( {rec_prof}) = {ts.machine.phi0[rec_turn]:.4f}\n'\
                f'Horizontal range (in pixels) of the region in '\
                  f'phase space of map elements:\n'\
                f' imin( {rec_prof}) =   {self.imin} and '\
                  f'imax( {rec_prof}) =  {self.imax}'

        print(out_s)


