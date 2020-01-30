'''Module containing the Tracking class

:Author(s): **Christoffer Hjertø Grindheim**
'''

import numpy as np
import sys
import logging

from ..utils import physics
from ..utils import assertions as asrt
from ..utils import exceptions as expt


class PhaseSpaceInfo:
    '''Class calculating and storing inforamtion about the reconstruction area.
    
    Parameters
    ----------
    machine: Machine
        Object containing machine parameters and settings. 

    Attributes
    ----------
    machine: Machine
        Object containing machine parameters and settings.
    jmin: ndarray, int
        Minimum energy for each phase of the phase space coordinate system.
    jmax: ndarray, int
        Maximum energy for each phase of the phase space coordinate system.
    imin: int
        Minimum phase, in phase space coordinates, of reconstruction area.
    imax: int
        Maximum phase, in phase space coordinates, of reconstruction area.
    dEbin: float
        Energy size of bins in phase space coordinate system.
    xorigin: float
        The absolute difference (in bins) between phase=0 and the origin
        of the reconstructed phase space coordinate system.
    '''
    def __init__(self, machine):
        self.machine = machine
        self.jmin = None
        self.jmax = None
        self.imin = None
        self.imax = None
        self.dEbin = None
        self.xorigin = None

    def find_binned_phase_energy_limits(self):
        '''This function finds the limits in the i (phase)
        and j (energy) axes of the reconstructed phase space coordinate system.

        The area within the limits of i and j will be the reconstruction area.
        This is where the particles will be populated.
        By setting the `full_pp_flag` to True in the provided `Machine` object,
        the reconstruction area will be set to the whole phase space image.

        This function gives a value to all attributes of the class.  
        
        Raises
        ------
        EnergyBinningError: Exception
            The energy size of the phase space coordinate
            bins are less than zero. 
        PhaseLimitsError: Exception
            Error in setting the limits in i (phase).
        EnergyLimitsError: Exception
            Error in setting the limits in j (energy).
        ArrayLengthError: Exception
            Difference in array shape of jmin and jmax.
        '''

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

        # Calculate limits (index of bins) in i-axis (phase axis),
        # 	adjust j-axis (energy axis)
        (jmin, jmax,
         self.imin,
         self.imax) = self._adjust_limits(jmax, jmin, imin, imax)

        self.jmin = jmin.astype(np.int32)
        self.jmax = jmax.astype(np.int32)
        
        # Ensuring that the output is valid
        self._assert_correct_arrays()

    def find_dEbin(self):
        '''Function to calculate the size of a energy bin in the
        reconstructed phase space coordinate system.
        
        Needed by
        :py:meth:`~tomo.tracking.phase_space_info.PhaseSpaceInfo\
.find_binned_phase_energy_limits`
        in order to calculate the reconstruction area.

        Returns
        -------
        dEbin: float
            Energy size of bins in phase space coordinate system.
        
        Raises
        ------
        EnergyBinningError: Exception
            The provided value of machine.demax is invalid.
        '''
        if self.xorigin is None:
            raise expt.EnergyBinningError(
                'xorigin must be calculated in order to find dEbin.')

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
                        / (self.machine.nbins - self.machine.synch_part_y))
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
                    / (self.machine.nbins - self.machine.synch_part_y))

    def calc_xorigin(self):
        '''Function for calculating xorigin.

        Needed by
        :py:meth:`~tomo.tracking.phase_space_info.PhaseSpaceInfo\
.find_binned_phase_energy_limits`
        in order to calculate the reconstruction area.

        Returns
        -------
        xorigin: float
        The absolute difference (in bins) between phase=0 and the origin
        of the reconstructed phase space coordinate system.
        '''
        beam_ref_turn = self.machine.beam_ref_frame * self.machine.dturns
        return (self.machine.phi0[beam_ref_turn]
                / (self.machine.h_num
                   * self.machine.omega_rev0[beam_ref_turn]
                   * self.machine.dtbin)
                - self.machine.synch_part_x)

    # Finding limits for distibuting particle over the full image
    # of the reconstructed phase space.
    def _limits_track_full_image(self):
        jmax = np.zeros(self.machine.nbins)
        jmin = np.copy(jmax)

        jmax[:] = self.machine.nbins
        jmin[:] = np.ceil(2.0 * self.machine.synch_part_y - jmax + 0.5)

        imin = np.int32(0)
        imax = np.int32(self.machine.nbins)
        return jmin, jmax, imin, imax

    # Finding limits for creating a smaller reconstruction area.
    def _limits_track_rec_area(self, dEbin):
        jmax = np.zeros(self.machine.nbins)
        jmin = np.copy(jmax)

        # Calculating turn and phases at the filmstart reference point.
        turn = self.machine.filmstart * self.machine.dturns
        phases = self._calculate_phases(turn)

        # Calculate the limits in energy at each phase 
        jmax = self._find_max_binned_energy(phases, turn, dEbin)
        jmin = self._find_min_binned_energy(jmax)

        # Find area in phase where energy limits are valid
        # (jmax - jmin >= 0).
        imin = self._find_min_binned_phase(jmin, jmax)
        imax = self._find_max_binned_phase(jmin, jmax)

        return jmin, jmax, imin, imax

    # Function for finding maximum energy (j max) for each bin in the profile.
    # The assumtion mad in the program is that jmin and jmax are mirrored.
    def _find_max_binned_energy(self, phases, turn, dEbin):

        energy = 0.0
        jmax_low = np.zeros(self.machine.nbins + 1)
        jmax_up = np.zeros(self.machine.nbins + 1)

        # finding max energy at edges of profiles
        for i in range(self.machine.nbins + 1):
            temp_energy = np.floor(self.machine.synch_part_y
                                   + self._trajectoryheight(
                                        phases[i], phases[0], energy, turn)
                                   / dEbin)
            
            jmax_low[i] = int(temp_energy)

            temp_energy = np.floor(self.machine.synch_part_y
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
        jmin = np.ceil(2.0 * self.machine.synch_part_y - jmax[:] - 0.5)
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
    # Calculates final limits of i-axis.
    def _adjust_limits(self, jmax, jmin, imin, imax):

        # Maximum and minimum bin, as spescified by user.
        max_dtbin = int(np.ceil(self.machine.max_dt / self.machine.dtbin))
        min_dtbin = int(self.machine.min_dt / self.machine.dtbin)

        if (min_dtbin > imin or self.machine.full_pp_flag):
            imin = min_dtbin
            jmax[:min_dtbin] = np.floor(self.machine.synch_part_y)
            jmin = np.ceil(2.0 * self.machine.synch_part_y - jmax + 0.5)

        if max_dtbin < imax or self.machine.full_pp_flag:
            imax = max_dtbin - 1 # -1 in order to count from idx 0
            jmax[max_dtbin:
                 self.machine.nbins] = np.floor(self.machine.synch_part_y)
            jmin = np.ceil(2.0 * self.machine.synch_part_y - jmax + 0.5)

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
