"""Module containing the ProgramsMachine class

:Author(s): **Anton Lu**
"""
import numpy as np
import scipy.constants as c
import logging
import typing as t
from scipy import optimize
from scipy import constants as cont

from ..utils import physics
from .machine_base import MachineABC
from .. import assertions as asrt
from .. import exceptions as ex

log = logging.getLogger(__name__)


class ProgramsMachine(MachineABC):
    """
    An alternative variant of the Machine class that is intended to be used
    with precomputed voltage, momentum, and phase programs.

    The voltage, momentum and phases are interpolated at each turn from the
    passed numpy arrays as opposed to linearly extrapolated or considered
    constant as in the :py:tomo.tracking.Machine:

    See superclass for documentation about inherited class variables.

    Parameters
    ----------
    voltage_function: np.ndarray
        A 2d array of shape (n_harmonics, :) containing the precomputed voltage
        program. The first row of the array must be the time stamps of the
        voltages in the second and third rows which contains the voltage
        amplitudes. 1 or 2 harmonics are supported.
    phase_program: np.ndarray
        A 2d array of shape (n_harmonics, :) containing the precomputed phase
        program. The first row of the array must be the time stamps of the
        phases in the second and third rows which contains the phase
        shifts. 1 or 2 harmonics are supported.
    momentum_program: np.ndarray
        A 2d array of shape (2, :) containing the precomputed momentum
        program. The first row of the array must be the time stamps of the
        momentum in the second row which contains the momentum values.
    harmonics: list
        A list of integers where each is the harmonic of an RF station.
        The number of harmonics must be the same as the passed voltage and
        phase functions.
    t_ref: float
        The cTime at which to the machine parameters are referenced.
        Must withing the time range defined in the voltage, phase and
        momentum programs.
    vat_now: bool
        Calculate values at turns immediately after initialisation.

    """

    def __init__(self,
                 dturns: int,
                 voltage_function: np.ndarray,
                 momentum_function: np.ndarray,
                 harmonics: t.List[int],
                 mean_orbit_rad: float, bending_rad: float,
                 trans_gamma: float, rest_energy: float,
                 nprofiles: int,
                 nbins: int,
                 dtbin: float,
                 t_ref: float,
                 phase_function: np.ndarray = None,
                 vat_now: bool = True,
                 **kwargs):
        asrt.assert_inrange(len(harmonics), 'harmonics', 1, 2,
                            ex.ArrayLengthError,
                            'Only 1 or 2 harmonics accepted.')
        super().__init__(dturns, mean_orbit_rad, bending_rad, trans_gamma,
                         rest_energy, nprofiles, nbins, dtbin, **kwargs)

        self.voltage_raw = voltage_function
        self.momentum_raw = momentum_function
        self.harmonics = harmonics

        if 'phi12' in kwargs:
            if phase_function is not None:
                raise ValueError('You can only pass either phi12 or '
                                 'phase_function, not both.')
            self.phase_raw = None
            self.process_phase = 0
        elif phase_function is not None:
            self.phase_raw = phase_function
            self.process_phase = 1
        else:
            log.debug('Neither phi12 nor phase program passed. Must compute '
                      'phi12 separately.')
            self.process_phase = 2

        self.circumference: float = 2 * np.pi * self.mean_orbit_rad
        self.t_ref = t_ref

        # init variables
        self.momentum_function: np.ndarray = None
        self.bdot: np.ndarray = None
        self.n_turns: int = None
        self.i0: int = None

        if vat_now:
            self.values_at_turns()

    def values_at_turns(self):
        """
        Calculated function values at each turn. The discrete momentum, voltage
        and phase programs are interpolated to each turn.
        """
        momentum_time, momentum_function = self._interpolate_momentum(
            self.momentum_raw[0, :],
            self.momentum_raw[1, :]
        )

        # interpolate first harmonic voltage & phase
        self.vrf1_at_turn = np.interp(
            momentum_time,  # same time steps as momentum program
            self.voltage_raw[0, :],  # voltage time
            self.voltage_raw[1, :]  # voltage values
        )

        if self.process_phase == 1:
            phase1 = np.interp(
                momentum_time,
                self.phase_raw[0, :],
                self.phase_raw[1, :]
            )
        elif self.process_phase == 0:
            if isinstance(self.phi12, np.ndarray):
                if self.phi12.size() == 1:
                    pass
                elif len(self.phi12.shape) == 2 and self.phi12.shape[1] == 2:
                    self.phi12 = np.interp(momentum_time, *self.phi12)
                elif len(self.phi12) == len(self.voltage_raw):
                    self.phi12 = np.interp(momentum_time, self.voltage_raw[0],
                                           self.phi12)
                else:
                    raise ValueError(f'Do not know how to interpolate phi12 '
                                     f'array of shape {self.phi12.shape}. '
                                     f'It should either match voltage '
                                     f'function or provide its own time axis.')

        # interpolate second harmonic voltage & phase if applicable
        if len(self.harmonics) == 2:
            self.vrf2_at_turn = np.interp(
                momentum_time,
                self.voltage_raw[0, :],
                self.voltage_raw[2, :]
            )
            self.h_ratio = self.harmonics[1] / self.harmonics[0]

            if self.process_phase == 1:
                phase2 = np.interp(momentum_time,
                                   self.phase_raw[0, :],
                                   self.phase_raw[2, :])
        else:
            self.vrf2_at_turn = np.zeros_like(self.vrf1_at_turn)
            phase2 = 0
            self.h_ratio = 1

        self.n_turns = len(momentum_time)

        momentum = momentum_function
        energy = np.sqrt(momentum ** 2 + self.e_rest ** 2)
        deltaE0 = np.diff(energy)
        gamma = np.sqrt(1 + (momentum / self.e_rest) ** 2)

        beta0 = np.sqrt(1 / (1 + (self.e_rest / momentum) ** 2))
        t_rev = np.dot(self.circumference, 1 / (beta0 * c.c))
        f_rev = 1 / t_rev
        time_at_turn = np.cumsum(t_rev)
        time_at_turn -= time_at_turn[self.i0]

        omega_rev0 = 2 * np.pi * f_rev
        if self.process_phase == 1:
            self.phi12 = (phase1 - phase2 + np.pi) / self.h_ratio
        elif self.process_phase == 2:
            self.phi12 = None

        eta0 = (1. - beta0 ** 2) - self.trans_gamma ** (-2)
        drift_coef = (2 * np.pi * self.h_num * eta0
                      / (energy * beta0 ** 2))

        bfield = momentum / (self.bending_rad * self.q) / cont.c
        bdot = np.gradient(bfield, time_at_turn)

        self.momentum_function = momentum_function

        self.bdot = bdot
        self.beta0 = beta0
        self.deltaE0 = deltaE0
        self.drift_coef = drift_coef
        self.e0 = energy
        self.eta0 = eta0
        self.omega_rev0 = omega_rev0
        self.phi0 = np.zeros_like(self.vrf1_at_turn)
        self.time_at_turn = time_at_turn

        if self.process_phase != 2:
            self.find_synchronous_phase()

    def calc_phi12(self, c04mode: int):
        """
        Calculates the phase difference between the two RF stations as seen
        by the particle.

        N.B. Do not forget to run find_synchronous_phase after this!

        Parameters
        ----------
        c04mode: int
            0, 1, or 2, representing
            * Bunch shortening mode
            * Bunch lengthening mode
            * h2 + 1 mode
            respectively
        """
        log.info(f'Calculating phi12 and overwriting previously calculated '
                 f'values.')

        self.phi12 = np.arcsin(2 * np.pi * self.bending_rad *
                               self.mean_orbit_rad *
                               self.bdot / self.vrf1_at_turn) \
            + np.pi / self.h_ratio

        if c04mode == 2:
            self.phi12 -= np.pi

    def find_synchronous_phase(self):

        if self.phi12 is None:
            raise ValueError('phi12 is None. You must pass a phase program, '
                             'phi12, or call the calc_phi12 method to '
                             'calculate phi12.')

        phi_lower, phi_upper = physics.find_phi_lower_upper(self, self.i0)

        self.phi0 = np.zeros_like(self.vrf1_at_turn)

        try:
            phi_start = optimize.newton(func=physics.rfvolt_rf1_pmch,
                                        x0=(phi_lower + phi_upper) / 2.0,
                                        fprime=physics.drfvolt_rf1_pmch,
                                        tol=0.0001,
                                        maxiter=100,
                                        args=(self, self.i0))
            synch_phase = optimize.newton(func=physics.rf_voltage_pmch,
                                          x0=phi_start,
                                          fprime=physics.drf_voltage_pmch,
                                          tol=0.0001,
                                          maxiter=100,
                                          args=(self, self.i0))
            self.phi0[self.i0] = synch_phase

            for i in range(self.i0 + 1, self.n_turns):
                self.phi0[i] = optimize.newton(func=physics.rf_voltage_pmch,
                                               x0=self.phi0[i - 1],
                                               fprime=physics.drf_voltage_pmch,
                                               tol=0.0001,
                                               maxiter=100,
                                               args=(self, i))
            for i in range(self.i0 - 1, -1, -1):
                self.phi0[i] = optimize.newton(func=physics.rf_voltage_pmch,
                                               x0=self.phi0[i + 1],
                                               fprime=physics.drf_voltage_pmch,
                                               tol=0.0001,
                                               maxiter=100,
                                               args=(self, i))
        except RuntimeError:
            raise ValueError('Could not find synchronous phase for supplied '
                             'parameters.')

    def _interpolate_momentum(self, time: np.ndarray, momentum: np.ndarray):
        """
        Interpolates momentum at each turn. The interpolated time array is
        then used to interpolating voltage and phase programs.

        The interpolation code is borrowed from CERN BLonD.

        Parameters
        ----------
        time: np.ndarray
            A 1d array with the time codes for each momentum value.
        momentum: np.ndarray
            A 1d array with momentum values, each corresponding to a timestamp.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Interpolated time and momentum at each turn.
        """

        beta_0 = np.sqrt(1 / (1 + (self.e_rest / momentum[0]) ** 2))
        T0 = self.circumference / (
                beta_0 * c.c)  # Initial revolution period [s]
        time_interp = [time[0] + T0]
        beta_interp = [beta_0]
        momentum_interp = [momentum[0]]

        # Interpolate data recursively
        time_interp.append(time_interp[-1]
                           + self.circumference / (beta_interp[0] * c.c))

        nturns = self.dturns * self.nprofiles
        self.i0 = self.machine_ref_frame * self.dturns
        i = 0
        turn = self.i0

        k = 0
        while turn < nturns:

            while time_interp[i + 1] <= time[k]:
                momentum_interp.append(
                    momentum[k - 1] + (momentum[k] - momentum[k - 1]) *
                    (time_interp[i + 1] - time[k - 1]) /
                    (time[k] - time[k - 1]))

                beta_interp.append(
                    np.sqrt(
                        1 / (1 + (self.e_rest / momentum_interp[i + 1]) ** 2)))

                time_interp.append(
                    time_interp[i + 1] + self.circumference / (
                            beta_interp[i + 1] * c.c))

                i += 1

                if time_interp[i + 1] > self.t_ref:
                    turn += 1

            k += 1

        time_interp.pop()
        time_interp = np.asarray(time_interp)
        beta_interp = np.asarray(beta_interp)
        momentum_interp = np.asarray(momentum_interp)

        # Cutting the input momentum on the desired cycle time
        if self.t_ref is not None:
            initial_index = np.min(np.where(time_interp >= self.t_ref)[0])
            initial_index -= self.i0
        else:
            initial_index = 0

        final_index = initial_index + nturns
        if final_index > len(time_interp) + 1:
            raise ValueError('Not enough data in momentum program to '
                             f'interpolate {nturns} turns.')

        momentum_time = time_interp[initial_index:final_index]
        momentum_function = momentum_interp[initial_index:final_index]

        return momentum_time, momentum_function
