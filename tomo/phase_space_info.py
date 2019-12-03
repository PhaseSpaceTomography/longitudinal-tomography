import numpy as np
import sys
import logging

from . import physics
from .utils import assertions as asrt
from .utils import exceptions as expt

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


class PhaseSpaceInfo:

    def __init__(self, machine):
        self.machine = machine
        self.jmin = None
        self.jmax = None
        self.imin = None
        self.imax = None
        self.dEbin = None
        self.xorigin = None


    # Main function for the class. finds limits in i (phase) and j (energy) axis.
    # variables:
    #   - turn_now: machine turn (=0 at profile = 0)
    #   - phases: phase at the edge of each bin along the i-axis
    def find_binned_phase_energy_limits(self):

        self.xorigin = self.calc_xorigin()
        self.dEbin = self.find_dEbin()

        # If dEbin is less than 0, an error is raised.  
        asrt.assert_greater(self.dEbin, 'dEbin', 0.0, expt.EnergyBinningError)

        # Is this still a valid choise with the new method?
        if self.machine.full_pp_flag == True:
            (jmin, jmax,
             imin, imax) = self._limits_track_full_image()
        else:
            (jmin, jmax,
             imin, imax) = self._limits_track_rec_area(self.dEbin)
        
        # self.allbin_min = allbin_min
        # self.allbin_max = allbin_max

        # Calculate limits (index of bins) in i-axis (phase axis),
        # 	adjust j-axis (energy axis)
        (jmin, jmax,
         self.imin,
         self.imax) = self._adjust_limits(jmax, jmin, imin, imax)

        self.jmin = jmin.astype(np.int32)
        self.jmax = jmax.astype(np.int32)
        
        # Ensuring that the output is valid
        self._assert_correct_arrays()

    # Calculate the size of bins in energy axis
    # Turn is at beam reference profile
    def find_dEbin(self):
        turn = self.machine.beam_ref_frame * self.machine.dturns    
        phases = self._calculate_phases(turn)
        delta_e_known = 0.0
        asrt.assert_not_equal(self.machine.demax, 'dEmax',
                              0.0, expt.EnergyBinningError,
                              'The specified maximum energy of '
                              'reconstructed phase space is invalid.')
        if self.machine.demax < 0.0:
            if physics.vrft(self.machine.vrf2,
                            self.machine.vrf2dot, turn) != 0.0:
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
                                  * physics.vrft(self.machine.vrf1,
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


    # Calculate the absolute difference (in bins) between phase=0 and
    # origin of the reconstructed phase-space coordinate system.
    def calc_xorigin(self):
        beam_ref_turn = self.machine.beam_ref_frame * self.machine.dturns
        return (self.machine.phi0[beam_ref_turn]
                / (self.machine.h_num
                   * self.machine.omega_rev0[beam_ref_turn]
                   * self.machine.dtbin)
                - self.machine.xat0)

    # Finding limits for tracking all pixels in reconstructed phase space.
    def _limits_track_full_image(self):
        jmax = np.zeros(self.machine.nbins)
        jmin = np.copy(jmax)

        jmax[:] = self.machine.nbins
        jmin[:] = np.ceil(2.0 * self.machine.yat0 - jmax + 0.5)

        imin = np.int32(0)
        imax = np.int32(self.machine.nbins)
        return jmin, jmax, imin, imax

    # Finding limits for tracking active pixels
    def _limits_track_rec_area(self, dEbin):
        jmax = np.zeros(self.machine.nbins)
        jmin = np.copy(jmax)

        # Calculating turn and phases at the filmstart reference point
        turn = self.machine.filmstart * self.machine.dturns
        phases = self._calculate_phases(turn)

        # Jmax to int already here? 
        jmax = self._find_max_binned_energy(phases, turn, dEbin)
        jmin = self._find_min_binned_energy(jmax)

        imin = self._find_min_binned_phase(jmin, jmax)
        imax = self._find_max_binned_phase(jmin, jmax)

        return jmin, jmax, imin, imax

    # Function for finding maximum energy (j max) for each bin in the profile
    def _find_max_binned_energy(self, phases, turn, dEbin):

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
    def _find_min_binned_energy(self, jmax, threshold=1):
        jmin = np.ceil(2.0 * self.machine.yat0 - jmax[:] - 0.5)
        return np.where(jmin[:] >= threshold, jmin[:], threshold)

    # Finding index for minimum phase for profile
    def _find_min_binned_phase(self, jmin, jmax):
        for i in range(0, self.machine.nbins):
            if jmax[i] - jmin[i] >= 0:
                return i

    # Finding index for maximum phase for profile
    def _find_max_binned_phase(self, jmin, jmax):
        for i in range(self.machine.nbins - 1, 0, -1):
            if jmax[i] - jmin[i] >= 0:
                return i

    # Adjustment of limits in relation to
    # 	specified input min/max index and found min/max in profile.
    # 	E.g. if profile_mini is greater than allbin_min, use profile_mini.
    # Calculates limits in i axis.
    def _adjust_limits(self, jmax, jmin, imin, imax):

        # Maximum and minimum bin, as spescified by user.
        max_dtbin = int(np.ceil(self.machine.max_dt / self.machine.dtbin))
        min_dtbin = int(self.machine.min_dt / self.machine.dtbin)

        if (min_dtbin > imin or self.machine.full_pp_flag):
            imin = min_dtbin
            jmax[:min_dtbin] = np.floor(self.machine.yat0)
            jmin = np.ceil(2.0 * self.machine.yat0 - jmax + 0.5)

        if max_dtbin < imax or self.machine.full_pp_flag:
            imax = max_dtbin - 1 # -1 in order to count from idx 0
            jmax[max_dtbin: self.machine.nbins] = np.floor(self.machine.yat0)
            jmin = np.ceil(2.0 * self.machine.yat0 - jmax + 0.5)

        return jmin, jmax, imin, imax

    # Returns an array of phases for a given turn
    def _calculate_phases(self, turn):
        indarr = np.arange(self.machine.nbins + 1)
        phases = ((self.xorigin + indarr)
                  * self.machine.dtbin
                  * self.machine.h_num
                  * self.machine.omega_rev0[turn])
        return phases

    # Trajectory height calculator
    def _trajectoryheight(self, phi, phi_known, delta_e_known, turn):
        cplx_height = 2.0 * self.machine.q / self.machine.drift_coef[turn]
        cplx_height *= (physics.vrft(self.machine.vrf1,
                                     self.machine.vrf1dot, turn)
                        * (np.cos(phi) - np.cos(phi_known))
                        + physics.vrft(self.machine.vrf2,
                                       self.machine.vrf2dot, turn)
                        * (np.cos(self.machine.h_ratio
                                  * (phi - self.machine.phi12))
                        - np.cos(self.machine.h_ratio
                                 * (phi_known - self.machine.phi12)))
                        / self.machine.h_ratio
                        + (phi - phi_known)
                        * physics.short_rf_voltage_formula(
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

    def _assert_correct_arrays(self):
        for film in range(self.machine.filmstart,
                          self.machine.filmstop,
                          self.machine.filmstep):

            # Testing imin and imax
            asrt.assert_inrange(self.imin, 'imin', 0, self.imax,
                                expt.PhaseLimitsError,
                                f'imin and imax out of bounds')
            asrt.assert_less_or_equal(self.imax, 'imax', self.jmax.size,
                                     expt.PhaseLimitsError,
                                     f'imin and imax out of bounds')

            # Testing jmin and jmax
            asrt.assert_array_in_range(self.jmin[self.imin:self.imax], 0,
                                       self.jmax[self.imin:self.imax],
                                       expt.EnergyLimitsError,
                                       msg=f'jmin and jmax out of bounds ',
                                       index_offset=self.imin)
            asrt.assert_array_less_eq(self.jmax[self.imin:self.imax],
                                      self.machine.nbins,
                                      expt.EnergyLimitsError,
                                      f'jmin and jmax out of bounds ')
            asrt.assert_equal(self.jmin.shape, 'jmin',
                              self.jmax.shape, expt.ArrayLengthError,
                              'jmin and jmax should have the same shape')
