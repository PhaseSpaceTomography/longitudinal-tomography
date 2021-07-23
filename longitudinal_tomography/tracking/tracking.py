"""Module containing the Tracking class.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
from typing import Tuple, TYPE_CHECKING, Callable

import numpy as np
import logging

from .. import assertions as asrt
from .__tracking import ParticleTracker
from ..cpp_routines import libtomo
from ..compat import fortran

if TYPE_CHECKING:
    from .machine_base import MachineABC
    from .machine import Machine

log = logging.getLogger(__name__)


class Tracking(ParticleTracker):
    """Class for particle tracking.

    This class perform the particle tracking based on the algorithm from
    the original tomography program. Here, an initial distribution of test
    particles are homogeneously distributed across the reconstruction area of
    the phase space image. Later, the particles will be tracked trough
    all machine turns and saved for every time frame.

    Parameters
    ----------
    machine: Machine
        Holds all information needed for the particle tracking and the
        generation of initial the particle distribution.

    Attributes
    ----------
    machine: Machine
        Holds all information needed for the particle tracking and the
        generation of initial the particle distribution.
    particles: Particles
        Creates and/or stores the initial particle distribution.
    nturns: int
        Number of machine turns particles should be tracked trough.
    self_field_flag: boolean
        Flag to indicate is self-fields should be included during tracking.
    fortran_flag: boolean
        Flag to indicate is a Fortran-style output should be printed to
        stdout during particle tracking.
    """

    def __init__(self, machine: 'MachineABC'):
        super().__init__(machine)

    def track(self, recprof: int, init_distr: Tuple[float, float] = None,
              callback: Callable = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Primary function for tracking particles.

        The tracking routine starts at a given time frame, with an initial
        distribution of particles. From here, the particles are tracked
        'forward' towards the last time frame and 'backwards' towards the
        first time frame.

        By default, an distribution of particles is spread out homogeneously
        over the area to be reconstructed. This area is found using the
        :class:`longitudinal_tomography.tracking.phase_space_info.PhaseSpaceInfo` class.
        The homogeneous distribution is placed on the time frame intended
        to be reconstructed for optimum quality. This is based on the
        original tomography algorithm.

        An user specified distribution can be given and override the
        default, automatic generation of particles.

        By calling
        :func:`longitudinal_tomography.tracking.__tracking.ParticleTracker.enable_self_fields`,
        a flag indicating that self-fields should be included is set.
        In this case, :func:`kick_and_drift_self` will be used.
        Note that tracking including self fields are much slower than without.

        By calling
        :func:`longitudinal_tomography.tracking.__tracking.ParticleTracker.enable_fortran_output`,
        an output resembling the original is written to stdout.
        Note that the values for the number of lost particles is **not valid**.
        Note also, that if the full Fortran output is to be printed,
        the automatic generation of particles must be performed.

        Parameters
        ----------
        recprof: int
            The profile (time frame) to be reconstructed. Here, the particles
            will have its initial distribution. By giving negative values as
            arguments, the index will count from the last time frame.
        init_distr: tuple, optional, default=None
            An user generated initial distribution. Must be given as a tuple of
            coordinates (dphi, denergy). dphi is the phase difference
            from the synchronous particle [rad]. denergy is the difference
            in energy from the synchronous particle [eV].
        callback: Callable
            Passing a callback with function signature
            (progress: int, total: int) will allow the tracking loop to call
            this function at the end of each turn, allowing the python caller
            to monitor the progress.

        Returns
        -------
        xp: ndarray
            2D array containing each particles x-coordinate at
            each time frame. Shape: (nprofiles, nparts).

            * If self-fields are enabled,
              the coordinates will be given as phase-space coordinates.
            * If self-fields are disabled, the returned x-coordinates will be
              given as phase [rad] relative to the synchronous particle.

        yp: ndarray, float
            2D array containing each particles y-coordinate at
            each time frame. Shape: (nprofiles, nparts).

            * If self-fields are enabled,
              the coordinates will be given as phase-space coordinates.
            * If self-fields are disabled, the returned y-coordinates will be
              given as energy [eV] relative to the synchronous particle.
        """

        recprof = asrt.assert_index_ok(
            recprof, self.machine.nprofiles, wrap_around=True)
        machine = self.machine

        if init_distr is None:
            # Homogeneous distribution is created based on the
            # original Fortran algorithm.
            log.info('Creating homogeneous distribution of particles.')
            self.particles.homogeneous_distribution(machine, recprof)
            coords = self.particles.coordinates_dphi_denergy

            # Print fortran style plot info. Needed for tomograph.
            if self.fortran_flag:
                print(fortran.write_plotinfo(
                    machine, self.particles, self._profile_charge))

        else:
            log.info('Using initial particle coordinates set by user.')
            self.particles.coordinates_dphi_denergy = init_distr
            coords = self.particles.coordinates_dphi_denergy

        dphi = coords[0]
        denergy = coords[1]

        rfv1 = machine.vrf1_at_turn * machine.q
        rfv2 = machine.vrf2_at_turn * machine.q

        # Tracking particles
        if self.self_field_flag:
            # Tracking with self-fields
            log.info('Tracking particles... (Self-fields enabled)')
            denergy = np.ascontiguousarray(denergy)
            dphi = np.ascontiguousarray(dphi)
            xp, yp = self.kick_and_drift_self(
                denergy, dphi, rfv1, rfv2, recprof)
        else:
            # Tracking without self-fields
            nparts = len(dphi)
            nturns = machine.dturns * (machine.nprofiles - 1)
            xp = np.zeros((machine.nprofiles, nparts))
            yp = np.zeros((machine.nprofiles, nparts))

            # Calling C++ implementation of tracking routine.
            libtomo.kick_and_drift(xp, yp, denergy, dphi, rfv1, rfv2,
                                   machine.phi0, machine.deltaE0,
                                   machine.drift_coef, machine.phi12,
                                   machine.h_ratio, machine.dturns,
                                   recprof, nturns, nparts,
                                   self.fortran_flag, callback=callback)

        log.info('Tracking completed!')
        return xp, yp

    def kick_and_drift(self, denergy: np.ndarray, dphi: np.ndarray,
                       rf1v: np.ndarray, rf2v: np.ndarray, rec_prof: int) -> \
            Tuple[np.ndarray, np.ndarray]:
        """Routine for tracking a distribution of particles for N turns.
        N is given by *tracking.nturns*

        A full C++ implementation is used in :func:`track`. This function is
        implemented as hybrid between Python and C++, and kept for reference.

        Parameters
        ----------
        denergy: ndarray
            particle energy relative to synchronous particle [eV]
        dphi: ndarray
            particle phase relative to synchronous particle [rad]
        rf1v: ndarray
            Radio frequency voltages (RF station 1),
            multiplied by particle charge
        rf2v: ndarray
            Radio frequency voltages (RF station 2),
            multiplied by particle charge
        rec_prof: int
            Time slice to initiate tracking.

        Returns
        -------
        dphi:
            Phase [rad] relative to the synchronous particle, for each particle
            at each measured time frame. Shape: (nprofiles, nparts).
        denergy: ndarray
            Energy [eV] relative to the synchronous particle, for each particle
            at each measured time slice [eV]. Shape: (nprofiles, nparts).
        """
        nparts = len(denergy)

        # Creating arrays for all tracked particles
        out_dphi = np.zeros((self.machine.nprofiles, nparts))
        out_denergy = np.copy(out_dphi)

        # Setting homogeneous coordinates to profile to be reconstructed.
        out_dphi[rec_prof] = np.copy(dphi)
        out_denergy[rec_prof] = np.copy(denergy)

        rec_turn = rec_prof * self.machine.dturns
        turn = rec_turn
        profile = rec_prof

        # Tracking 'upwards'
        while turn < self.nturns:
            # Calculating change in phase for each particle at a turn
            dphi = libtomo.drift(denergy, dphi, self.machine.drift_coef,
                                 nparts, turn)
            turn += 1
            # Calculating change in energy for each particle at a turn
            denergy = libtomo.kick(self.machine, denergy, dphi, rf1v, rf2v,
                                   nparts, turn)

            if turn % self.machine.dturns == 0:
                profile += 1
                out_dphi[profile] = np.copy(dphi)
                out_denergy[profile] = np.copy(denergy)
                if self.fortran_flag:
                    fortran.print_tracking_status(rec_prof, profile)

        # Starting again from homogenous distribution
        dphi = np.copy(out_dphi[rec_prof])
        denergy = np.copy(out_denergy[rec_prof])
        turn = rec_turn
        profile = rec_prof

        # Tracking 'downwards'
        while turn > 0:
            # Calculating change in energy for each particle at a turn
            denergy = libtomo.kick(self.machine, denergy, dphi, rf1v, rf2v,
                                   nparts, turn, up=False)
            turn -= 1
            # Calculating change in phase for each particle at a turn
            dphi = libtomo.drift(denergy, dphi, self.machine.drift_coef,
                                 nparts, turn, up=False)

            if turn % self.machine.dturns == 0:
                profile -= 1
                out_dphi[profile] = np.copy(dphi)
                out_denergy[profile] = np.copy(denergy)
                if self.fortran_flag:
                    fortran.print_tracking_status(rec_prof, profile)

        return out_dphi, out_denergy

    def kick_and_drift_self(self, denergy: np.ndarray, dphi: np.ndarray,
                            rf1v: np.ndarray, rf2v: np.ndarray,
                            rec_prof: int) -> Tuple[np.ndarray, np.ndarray]:
        """Routine for tracking a given distribution of particles,\
        including self-fields. Implemented as hybrid between Python and C++.

        Routine for tracking, with self-fields, a distribution of
        particles for N turns. N is given by *tracking.nturns*.

        Used by the function
        :py:meth:`longitudinal_tomography.tracking.tracking.Tracking.track`
        to track using self-fields.

        Returns the coordinates of the particles as phase space coordinates
        for efficiency reasons due to tracking algorithm based on the
        Fortran tomography. large room for improvement.

        Fortran output not yet supported.

        Parameters
        ----------
        denergy: ndarray
            particle energy relative to synchronous particle [eV]
        dphi: ndarray
            particle phase relative to synchronous particle [rad]
        rf1v: ndarray
            Radio frequency voltages (RF station 1),
            multiplied by particle charge
        rf2v: ndarray
            Radio frequency voltages (RF station 2),
            multiplied by particle charge
        rec_prof: int
            Time slice to initiate tracking.

        Returns
        -------
        xp: ndarray, float
            2D array holding the x-coordinates of each particles at
            each time frame (nprofiles, nparts). Coordinates given
            in bins of phase space coordinate system.
        yp: ndarray, float
            2D array holding the y-coordinates of each particles at
            each time frame (nprofiles, nparts). Coordinates given
            in bins of phase space coordinate system.
        """
        nparts = len(denergy)

        # Creating arrays for all tracked particles
        xp = np.zeros((self.machine.nprofiles, nparts))
        yp = np.copy(xp)

        rec_turn = rec_prof * self.machine.dturns
        turn = rec_turn
        profile = rec_prof

        # To be saved for downwards tracking
        dphi0 = np.copy(dphi)
        denergy0 = np.copy(denergy)

        xp[rec_prof] = self._calc_xp_sf(
            dphi, self.machine.phi0[rec_turn],
            self.particles.xorigin, self.machine.h_num,
            self.machine.omega_rev0[rec_turn],
            self.machine.dtbin, self._phiwrap)

        yp[rec_prof] = (denergy / self.particles.dEbin
                        + self.machine.synch_part_y)

        if self.fortran_flag:
            fortran.print_tracking_status(rec_prof, profile)
        while turn < self.nturns:
            dphi = libtomo.drift(denergy, dphi, self.machine.drift_coef,
                                 nparts, turn)

            turn += 1

            temp_xp = self._calc_xp_sf(dphi, self.machine.phi0[turn],
                                       self.particles.xorigin,
                                       self.machine.h_num,
                                       self.machine.omega_rev0[turn],
                                       self.machine.dtbin,
                                       self._phiwrap)
            selfvolt = self._vself[profile, temp_xp.astype(int) - 1]

            denergy = libtomo.kick(self.machine, denergy, dphi, rf1v, rf2v,
                                   nparts, turn)
            denergy += selfvolt * self.machine.q

            if turn % self.machine.dturns == 0:
                profile += 1
                xp[profile] = temp_xp
                yp[profile] = (denergy / self.particles.dEbin
                               + self.machine.synch_part_y)
                if self.fortran_flag:
                    fortran.print_tracking_status(rec_prof, profile)

        dphi = dphi0
        denergy = denergy0
        turn = rec_turn
        profile = rec_prof - 1
        temp_xp = xp[rec_prof]

        while turn > 0:
            selfvolt = self._vself[profile, temp_xp.astype(int)]

            denergy = libtomo.kick(self.machine, denergy, dphi, rf1v, rf2v,
                                   nparts, turn, up=False)

            denergy += selfvolt * self.machine.q

            turn -= 1

            dphi = libtomo.drift(denergy, dphi, self.machine.drift_coef,
                                 nparts, turn, up=False)

            temp_xp = self._calc_xp_sf(
                dphi, self.machine.phi0[turn], self.particles.xorigin,
                self.machine.h_num, self.machine.omega_rev0[turn],
                self.machine.dtbin, self._phiwrap)

            if turn % self.machine.dturns == 0:
                xp[profile] = temp_xp
                yp[profile] = (denergy / self.particles.dEbin
                               + self.machine.synch_part_y)
                profile -= 1
                if self.fortran_flag:
                    fortran.print_tracking_status(rec_prof, profile)

        return xp, yp

    # Calculate from physical coordinates to x-coordinates.
    # Needed for tracking using self-fields.
    # TODO: removed njit, reimplement in C in the future
    @staticmethod
    def _calc_xp_sf(dphi: np.ndarray, phi0: np.ndarray, xorigin: int, h_num,
                    omega_rev0: np.ndarray, dtbin: int, phiwrap: float) \
            -> np.ndarray:
        temp_xp = (dphi + phi0 - xorigin * h_num * omega_rev0 * dtbin)
        temp_xp = ((temp_xp - phiwrap * np.floor(temp_xp / phiwrap))
                   / (h_num * omega_rev0 * dtbin))
        return temp_xp
