import numpy as np
import scipy.constants as c
import logging
import typing as t

from .machine_base import MachineABC
from .. import assertions as asrt
from .. import exceptions as ex


log = logging.getLogger(__name__)


class ProgramsMachine(MachineABC):
    """
    Machine class intended to be used with precomputed voltage, momentum, and
    phase programs
    """
    def __init__(self,
                 dturns: int,
                 voltage_function: np.ndarray,
                 phase_function: np.ndarray,
                 momentum_function: np.ndarray,
                 harmonics: t.List[int],
                 mean_orbit_rad: float, bending_rad: float,
                 trans_gamma: float, rest_energy: float,
                 n_profiles: int,
                 n_bins: int,
                 dtbin: float,
                 t_start: float = None,
                 t_end: float = None,
                 vat_now: bool = True,
                 **kwargs):
        asrt.assert_inrange(len(harmonics), 'harmonics', 1, 2,
                            ex.ArrayLengthError,
                            'Only 1 or 2 harmonics accepted.')
        kwargs['h_num'] = harmonics[0]
        super().__init__(dturns, mean_orbit_rad, bending_rad, trans_gamma,
                         rest_energy, n_profiles, n_bins, dtbin, **kwargs)

        self.voltage_raw = voltage_function
        self.phase_raw = phase_function
        self.momentum_raw = momentum_function
        self.harmonics = harmonics

        self.circumference = 2 * np.pi * self.mean_orbit_rad
        self.t_start: float = t_start
        self.t_end: float = t_end

        self.flat_bottom = 0
        self.flat_top = 0

        # init variables
        self.momentum_time: np.ndarray = None
        self.momentum_function: np.ndarray = None
        self.voltage_time: np.ndarray = None
        self.voltage_function: np.ndarray = None
        self.phase_function: np.ndarray = None
        self.phase_time: np.ndarray = None
        self.n_turns: int = None

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

        self.vrf1_at_turn = np.interp(
            momentum_time,  # same time steps as momentum program
            self.voltage_raw[0, :],  # voltage time
            self.voltage_raw[1, :]   # voltage values
        )
        phase1 = np.interp(momentum_time,
                           self.phase_raw[0, :],
                           self.phase_raw[1, :])

        if len(self.harmonics) == 2:
            self.vrf2_at_turn = np.interp(
                momentum_time,
                self.voltage_raw[0, :],
                self.voltage_raw[2, :]
            )
            self.h_ratio = self.harmonics[1] / self.harmonics[0]
            phase2 = np.interp(momentum_time,
                               self.phase_raw[0, :],
                               self.phase_raw[2, :])
        else:
            self.vrf2_at_turn = np.zeros_like(self.vrf1_at_turn)
            phase2 = 0
            self.h_ratio = 1

        self.n_turns = len(momentum_time)
        self.h_num = self.harmonics[0]
        # self.dturns = int(np.round(self.n_turns / self.nprofiles))

        momentum = momentum_function
        self.momentum_time = momentum_time
        self.momentum_function = momentum_function
        energy = np.sqrt(momentum**2 + self.e_rest**2)
        gamma = np.sqrt(1 + (momentum / self.e_rest) ** 2)

        self.beta0 = np.sqrt(1/(1 + (self.e_rest/momentum)**2))
        t_rev = np.dot(self.circumference, 1/(self.beta0*c.c))
        f_rev = 1/t_rev

        self.deltaE0 = np.diff(energy)
        self.omega_rev0 = 2*np.pi*f_rev
        self.phi12 = (phase1 - phase2 + np.pi) / self.h_ratio
        self.time_at_turn = np.cumsum(t_rev)

        momentum_compaction = 1 / self.trans_gamma**2
        # self.eta0 = momentum_compaction - np.power(gamma, -2.)
        self.e0 = energy
        self.eta0 = (1. - self.beta0**2) - self.trans_gamma**(-2)
        self.drift_coef = (2 * np.pi * self.harmonics[0] * self.eta0
                           / (energy * self.beta0 ** 2))

        denergy = np.append(self.deltaE0, self.deltaE0[-1])
        acceleration_ratio = denergy/(self.q*self.vrf1_at_turn)
        acceleration_test = np.where((acceleration_ratio > -1) *
                                     (acceleration_ratio < 1) is False)[0]

        # Validity check on acceleration_ratio
        if acceleration_test.size > 0:
            log.warning('WARNING in calculate_phi_s(): acceleration is not '
                        'possible (momentum increment is too big or voltage too '
                        'low) at index {}'.format(acceleration_test))

        phi_s = np.arcsin(acceleration_ratio)

        # Identify where eta swaps sign
        eta0 = -self.eta0
        eta0_middle_points = (eta0[1:] + eta0[:-1])/2
        eta0_middle_points = np.append(eta0_middle_points, eta0[-1])
        index = np.where(eta0_middle_points > 0)[0]
        index_below = np.where(eta0_middle_points < 0)[0]

        # Project phi_s in correct range
        phi_s[index] = (np.heaviside(np.sign(self.q),0) * np.pi - phi_s[index]) % (2*np.pi)
        phi_s[index_below] = (np.heaviside(np.sign(self.q),0) * np.pi + phi_s[index_below]) \
                             % (2*np.pi)
        self.phi0 = phi_s - np.pi

    def _interpolate_momentum(self, time: np.ndarray, momentum: np.ndarray):
        # code stolen from blond

        beta_0 = np.sqrt(1 / (1 + (self.e_rest / momentum[0]) ** 2))
        T0 = self.circumference / (beta_0 * c.c)  # Initial revolution period [s]
        shift = time[0] - self.flat_bottom * T0
        time_interp = shift + T0 * np.arange(0, self.flat_bottom + 1)
        beta_interp = beta_0 * np.ones(self.flat_bottom + 1)
        momentum_interp = momentum[0] * np.ones(self.flat_bottom + 1)

        time_interp = time_interp.tolist()
        beta_interp = beta_interp.tolist()
        momentum_interp = momentum_interp.tolist()

        time_start_ramp = np.max(time[momentum == momentum[0]])
        time_end_ramp = np.min(time[momentum == momentum[-1]])

        # Interpolate data recursively
        time_interp.append(time_interp[-1]
                           + self.circumference / (beta_interp[0] * c.c))

        i = self.flat_bottom

        if self.t_start is not None:
            initial_index = np.min(np.where(time >= self.t_start)[0])
        else:
            initial_index = 0
        if self.t_end is not None:
            final_index = np.max(np.where(time <= self.t_end)[0])+1
        else:
            final_index = len(time)

        for k in range(initial_index, final_index):

            while time_interp[i + 1] <= time[k]:
                momentum_interp.append(
                    momentum[k - 1] + (momentum[k] - momentum[k - 1]) *
                    (time_interp[i + 1] - time[k - 1]) /
                    (time[k] - time[k - 1]))

                beta_interp.append(
                    np.sqrt(
                        1 / (1 + (self.e_rest / momentum_interp[i + 1]) ** 2)))

                time_interp.append(
                    time_interp[i + 1] + self.circumference / (beta_interp[i + 1] * c.c))

                i += 1

        time_interp.pop()
        time_interp = np.asarray(time_interp)
        beta_interp = np.asarray(beta_interp)
        momentum_interp = np.asarray(momentum_interp)

        # Obtain flat top data, extrapolate to constant
        if self.flat_top > 0:
            time_interp = np.append(
                time_interp,
                time_interp[-1] + self.circumference*np.arange(1, self.flat_top+1)
                / (beta_interp[-1]*c.c))

            beta_interp = np.append(
                beta_interp, beta_interp[-1]*np.ones(self.flat_top))

            momentum_interp = np.append(
                momentum_interp,
                momentum_interp[-1]*np.ones(self.flat_top))

        # Cutting the input momentum on the desired cycle time
        if self.t_start is not None:
            initial_index = np.min(np.where(time_interp >= self.t_start)[0])
        else:
            initial_index = 0
        # if self.t_end is not None:
        #     final_index = np.max(np.where(time_interp <= self.t_end)[0])+1
        # else:
        #     final_index = len(time_interp)
        final_index = min(self.dturns * self.nprofiles + initial_index,
                             len(time_interp))

        momentum_time = time_interp[initial_index:final_index]
        momentum_function = momentum_interp[initial_index:final_index]

        return momentum_time, momentum_function
