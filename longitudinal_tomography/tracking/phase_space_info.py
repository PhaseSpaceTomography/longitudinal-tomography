"""Module containing the PhaseSpaceInfo class

:Author(s): **Christoffer Hjertø Grindheim**
"""
from typing import TYPE_CHECKING, Tuple

import numpy as np

from ..exceptions import EnergyBinningError, EnergyLimitsError, \
    PhaseLimitsError, ArrayLengthError
from .. import assertions as asrt
from ..utils import physics

if TYPE_CHECKING:
    from .machine import Machine


class PhaseSpaceInfo:
    """Class for calculating and storing information about the phase
    space reconstruction area.

    Parameters
    ----------
    machine: Machine
        Object containing machine parameters and settings.

    Attributes
    ----------
    machine: Machine
        Object containing machine parameters and settings.
    jmin: ndarray
        1D array of integers holding the minimum energy for each phase
        of the phase space coordinate system.
    jmax: ndarray
        1D array of integers holding the maximum energy for each phase
        of the phase space coordinate system.
    imin: int
        Minimum phase of reconstruction area, given in phase space coordinates.
    imax: int
        Maximum phase of reconstruction area, given in phase space coordinates.
    dEbin: float
        Energy size of bins in phase space coordinate system.
    xorigin: float
        The absolute difference (in bins) between phase=0 and the origin
        of the reconstructed phase space coordinate system.
    """

    def __init__(self, machine: 'Machine'):
        self.machine = machine
        self.jmin = None
        self.jmax = None
        self.imin = None
        self.imax = None
        self.dEbin = None
        self.xorigin = None

    def find_binned_phase_energy_limits(self):
        """This function finds the limits in the i (phase) and j (energy) axes
        of the reconstructed phase space coordinate system.

        The area within the limits of i and j will be the phase space
        reconstruction area. This is where the particles will be populated.
        By setting the `full_pp_flag` in the provided
        :class:`~longitudinal_tomography.tracking.machine.Machine`
        object to True, the reconstruction area will be set to all of the
        phase space image.

        This function gives a value to all attributes of the class.

        Raises
        ------
        EnergyBinningError: Exception
            Size of each phase space bin in the energy axis is less than zero.
        PhaseLimitsError: Exception
            Error in the limits of the i (phase) axis.
        EnergyLimitsError: Exception
            Error in the limits of the j (energy) axis.
        ArrayLengthError: Exception
            Difference in length of jmin and jmax.
        """

        self.xorigin = self.calc_xorigin()
        self.dEbin = self.find_dEbin()

        # If dEbin is less than 0, an error is raised.
        asrt.assert_greater(self.dEbin, 'dEbin', 0.0, EnergyBinningError)

        # Is this still a valid choice with the new method?
        if self.machine.full_pp_flag is True:
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
        self._assert_correct_ijlimits()

    def find_dEbin(self) -> np.float64:
        """Function to calculate the size of a energy bin in the
        reconstructed phase space coordinate system.

        Needed by
        :meth:`longitudinal_tomography.tracking.phase_space_info.PhaseSpaceInfo.find_binned_phase_energy_limits`
        in order to calculate the reconstruction area.

        Returns
        -------
        dEbin:
            Energy size of bins in phase space coordinate system.

        Raises
        ------
        EnergyBinningError: Exception
            The provided value of machine.demax is invalid, or xorigin is
            not calculated before the function is called.
        """
        if self.xorigin is None:
            raise EnergyBinningError(
                'xorigin must be calculated in order to find dEbin.')

        turn = self.machine.beam_ref_frame * self.machine.dturns
        phases = self._calculate_phases(turn)
        delta_e_known = 0.0
        asrt.assert_not_equal(self.machine.demax, 'dEmax',
                              0.0, EnergyBinningError,
                              'The specified maximum energy of '
                              'reconstructed phase space is invalid.')
        if self.machine.demax < 0.0:
            if self.machine.vrf2_at_turn[turn] != 0.0:
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
                                  * self.machine.vrf1_at_turn[turn]
                                  * np.cos(self.machine.phi0[turn])
                                  / (2 * np.pi * self.machine.h_num
                                     * self.machine.eta0[turn]))
                        * self.machine.dtbin
                        * self.machine.h_num
                        * self.machine.omega_rev0[turn])
        else:
            return (float(self.machine.demax)
                    / (self.machine.nbins - self.machine.synch_part_y))

    def calc_xorigin(self) -> np.float64:
        """Function for calculating xorigin.

        Needed by
        :func:`~longitudinal_tomography.tracking.phase_space_info.PhaseSpaceInfo.find_binned_phase_energy_limits`
        in order to calculate the reconstruction area.

        Returns
        -------
        xorigin: float
            The absolute difference (in bins) between phase=0 and the origin
            of the reconstructed phase space coordinate system.
        """
        beam_ref_turn = self.machine.beam_ref_frame * self.machine.dturns
        return (self.machine.phi0[beam_ref_turn]
                / (self.machine.h_num
                   * self.machine.omega_rev0[beam_ref_turn]
                   * self.machine.dtbin)
                - self.machine.synch_part_x)

    # Finding limits for distributing particle over the full image
    # of the reconstructed phase space.
    def _limits_track_full_image(self) \
            -> Tuple[np.ndarray, np.ndarray, np.int32, np.int32]:
        jmax = np.zeros(self.machine.nbins)
        jmin = np.copy(jmax)

        jmax[:] = self.machine.nbins
        jmin[:] = np.ceil(2.0 * self.machine.synch_part_y - jmax + 0.5)

        imin = np.int32(0)
        imax = np.int32(self.machine.nbins)
        return jmin, jmax, imin, imax

    # Finding limits for creating a smaller reconstruction area.
    def _limits_track_rec_area(self, dEbin) \
            -> Tuple[np.ndarray, np.ndarray, int, int]:
        jmax = np.zeros(self.machine.nbins)
        jmin = np.copy(jmax)

        # Calculating turn and phases at the filmstart reference point.
        turn = self.machine.filmstart * self.machine.dturns
        phases = self._calculate_phases(turn)

        # Calculate the limits in energy at each phase
        jmax = self._find_max_binned_energy(phases, turn, dEbin)
        jmin = self._find_min_binned_energy(jmax)
        self._assert_jlimits_ok(jmin, jmax)

        # Find area in phase where energy limits are valid
        # (jmax - jmin >= 0).
        imin = self._find_min_binned_phase(jmin, jmax)
        imax = self._find_max_binned_phase(jmin, jmax)

        return jmin, jmax, imin, imax

    # Function for finding maximum energy (j max) for each bin in the profile.
    # The assumption mad in the program is that jmin and jmax are mirrored.
    def _find_max_binned_energy(self, phases: np.ndarray, turn: int,
                                dEbin: np.float64):

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
    def _find_min_binned_energy(self, jmax: np.ndarray, threshold: int = 1) \
            -> np.ndarray:
        jmin = np.ceil(2.0 * self.machine.synch_part_y - jmax[:] - 0.5)
        return np.where(jmin[:] >= threshold, jmin[:], threshold)

    # Finding index for minimum phase for profile
    def _find_min_binned_phase(
            self, jmin: np.ndarray, jmax: np.ndarray) -> int:
        for i in range(0, self.machine.nbins):
            if jmax[i] - jmin[i] >= 0:
                return i

    # Finding index for maximum phase for profile
    def _find_max_binned_phase(
            self, jmin: np.ndarray, jmax: np.ndarray) -> int:
        for i in range(self.machine.nbins - 1, 0, -1):
            if jmax[i] - jmin[i] >= 0:
                return i

    # Adjustment of limits in relation to
    # 	specified input min/max index and found min/max in profile.
    # 	E.g. if profile_mini is greater than allbin_min, use profile_mini.
    # Calculates final limits of i-axis.
    def _adjust_limits(self, jmax: np.ndarray, jmin: np.ndarray,
                       imin: int, imax: int) \
            -> Tuple[np.ndarray, np.ndarray, int, int]:

        # Maximum and minimum bin, as specified by user.
        max_dtbin = int(np.ceil(self.machine.max_dt / self.machine.dtbin))
        min_dtbin = int(self.machine.min_dt / self.machine.dtbin)

        if min_dtbin > imin or self.machine.full_pp_flag:
            imin = min_dtbin
            jmax[:min_dtbin] = np.floor(self.machine.synch_part_y)
            jmin = np.ceil(2.0 * self.machine.synch_part_y - jmax + 0.5)

        if max_dtbin < imax or self.machine.full_pp_flag:
            imax = max_dtbin - 1  # -1 in order to count from idx 0
            jmax[max_dtbin:
                 self.machine.nbins] = np.floor(self.machine.synch_part_y)
            jmin = np.ceil(2.0 * self.machine.synch_part_y - jmax + 0.5)

        return jmin, jmax, imin, imax

    # Returns an array of phases for a given turn
    def _calculate_phases(self, turn: int) -> np.ndarray:
        indarr = np.arange(self.machine.nbins + 1)
        phases = ((self.xorigin + indarr)
                  * self.machine.dtbin
                  * self.machine.h_num
                  * self.machine.omega_rev0[turn])
        return phases

    # Trajectory height calculator
    def _trajectoryheight(self, phi: np.ndarray, phi_known: float,
                          delta_e_known: float, turn: int) -> float:
        machine = self.machine
        if isinstance(machine.phi12, np.ndarray):
            phi12 = machine.phi12[turn]
        else:
            phi12 = machine.phi12

        cplx_height = 2.0 * machine.q / machine.drift_coef[turn]
        cplx_height *= (machine.vrf1_at_turn[turn]
                        * (np.cos(phi) - np.cos(phi_known))
                        + machine.vrf2_at_turn[turn]
                        * (np.cos(machine.h_ratio
                                  * (phi - phi12))
                           - np.cos(machine.h_ratio
                                    * (phi_known - phi12)))
                        / machine.h_ratio
                        + (phi - phi_known)
                        * physics.rf_voltage_at_phase(
                    machine.phi0[turn],
                    machine.vrf1_at_turn[turn],
                    machine.vrf2_at_turn[turn],
                    machine.h_ratio, phi12))
        cplx_height += delta_e_known ** 2

        if np.size(cplx_height) > 1:
            # Returning array
            cplx_height = np.array(cplx_height, dtype=complex)
            cplx_height = np.sqrt(cplx_height)
        else:
            # Returning scalar
            cplx_height = np.sqrt(complex(cplx_height))

        return cplx_height.real

    def _assert_correct_ijlimits(self):
        # Testing imin and imax
        asrt.assert_inrange(self.imin, 'imin', 0, self.imax,
                            PhaseLimitsError,
                            f'imin and imax out of bounds')
        asrt.assert_less_or_equal(self.imax, 'imax', self.jmax.size,
                                  PhaseLimitsError,
                                  f'imin and imax out of bounds')
        # Testing jmin and jmax
        asrt.assert_array_in_range(self.jmin[self.imin:self.imax], 0,
                                   self.jmax[self.imin:self.imax],
                                   EnergyLimitsError,
                                   msg=f'jmin and jmax out of bounds ',
                                   index_offset=self.imin)
        asrt.assert_array_less_eq(self.jmax[self.imin:self.imax],
                                  self.machine.nbins,
                                  EnergyLimitsError,
                                  f'jmin and jmax out of bounds ')
        asrt.assert_equal(self.jmin.shape, 'jmin',
                          self.jmax.shape, ArrayLengthError,
                          'jmin and jmax should have the same shape')
        self._assert_jlimits_ok(self.jmin, self.jmax)

    # Testing that there is a difference between jmin and jmax
    def _assert_jlimits_ok(self, jmin: np.ndarray, jmax: np.ndarray):
        if all(jmin >= jmax):
            raise EnergyLimitsError(
                'All of jmin is larger than or equal to jmax; '
                'the size of the reconstruction area is zero.')
