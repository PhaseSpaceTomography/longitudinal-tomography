"""Module containing fundamental physics formulas.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
import typing as t
from numbers import Number
from collections.abc import Sequence

import numpy as np
from multipledispatch import dispatch
from scipy import optimize, constants

if t.TYPE_CHECKING:
    from ..tracking.machine import Machine
    from ..tracking.programs_machine import ProgramsMachine
    from ..tracking.machine_base import MachineABC


arr_or_float = t.Union[np.ndarray, float]


def b_to_e(machine: 'Machine') -> float:
    """
    Calculates the energy for a particle
    in a circular machine at dipole field B.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.

    Returns
    -------
    energy: float
        Energy calculated from B-field.
    """
    return np.sqrt((machine.q
                    * machine.b0
                    * machine.bending_rad
                    * constants.c) ** 2
                   + machine.e_rest ** 2)


def lorentz_beta(machine: 'MachineABC', rf_turn: int) -> float:
    """Calculates Lorentz beta factor (v/c) at a turn.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.
    rf_turn: int
        At which RF turn the calculation should happen.

    Returns
    -------
    Lorentz beta: float
        The Lorenz beta factor (v/c) at a given turn.
    """
    return np.sqrt(1.0 - (float(machine.e_rest) /
                          machine.e0[rf_turn]) ** 2)


def rfvolt_rf1(phi: float, v1: float, bdot: float, radius: float,
               bending_rad: float, charge: float) -> float:
    """Calculates RF voltage1 seen by particle.
    Needed by the Newton root finder to calculate phi0.

    Parameters
    ----------
    phi: float
        Phase where voltage is to be calculated.
    v1: np.ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.
    bdot: float
        Time derivative of B-field (considered constant) [T/s].
    radius: float
        Mean orbit radius of machine [m].
    bending_rad: float
        Machine bending radius [m].
    charge: float
        Charge state of accelerated particle.

    Returns
    -------
    voltage: float
        Voltage from RF system 1, as seen by particle.
    """
    q_sign = np.sign(charge)
    return (v1 * np.sin(phi)
            - 2 * np.pi * radius
            * bending_rad * bdot * q_sign)


def drfvolt_rf1(phi: arr_or_float, v1: arr_or_float) -> arr_or_float:
    """Calculates derivative of RF voltage1 seen by particle.
    Needed by the Newton root finder to calculate phi0.

    Parameters
    ----------
    phi: float
        Phase where voltage is to be calculated.
    v1: float or np.ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.

    Returns
    -------
    Derivative of voltage: float
        Time-derivative of voltage from RF system 1, as seen by particle.
    """
    return v1 * np.cos(phi)


def rfvolt_rf1_mch(phi: float, machine: 'Machine', rf_turn: int) -> float:
    """Objective function used in phi0 Newton optimization"""
    turn_time = machine.time_at_turn[rf_turn]
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)

    return rfvolt_rf1(phi, v1, machine.bdot, machine.mean_orbit_rad,
                      machine.bending_rad, machine.q)


def drfvolt_rf1_mch(phi: float, machine: 'Machine', rf_turn: int) -> float:
    """Objective function used in phi0 Newton optimization"""
    turn_time = machine.time_at_turn[rf_turn]
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    return drfvolt_rf1(phi, v1)


def rfvolt_rf1_pmch(phi: float, machine: 'ProgramsMachine', rf_turn: int) \
        -> float:
    """Objective function used in phi0 Newton optimization"""
    v1 = machine.vrf1_at_turn[rf_turn]
    bdot = machine.bdot[rf_turn]

    return rfvolt_rf1(phi, v1, bdot, machine.mean_orbit_rad,
                      machine.bending_rad, machine.q)


def drfvolt_rf1_pmch(phi: float, machine: 'ProgramsMachine', rf_turn: int) \
        -> float:
    """Objective function used in phi0 Newton optimization"""
    v1 = machine.vrf1_at_turn[rf_turn]
    return drfvolt_rf1(phi, v1)


def rf_voltage(phi: arr_or_float, v1: arr_or_float, v2: arr_or_float,
               bdot: arr_or_float, h_ratio: float, phi12: arr_or_float,
               radius: float, bending_rad: float, charge: float) \
        -> arr_or_float:
    """Calculates RF voltage from both RF systems seen by particle.
    Needed by the Newton root finder to calculate phi0.

    Parameters
    ----------
    phi: float
        Phase where voltage is to be calculated.
    v1: np.ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.
    v2: np.ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.
    bdot: float
        Time derivative of B-field (considered constant) [T/s].
    h_ratio: float
        Ratio of harmonics between the two RF systems.
    phi12: float or np.ndarray
        Phase difference between the two RF systems, turn-by-turn, or
        considered constant.
    radius: float
        Mean orbit radius of machine [m].
    bending_rad: float
        Machine bending radius [m].
    charge: float
        Charge state of accelerated particle.

    Returns
    -------
    voltage: float
        Voltage from both RF systems, as seen by particle.
    """
    q_sign = np.sign(charge)
    return (v1 * np.sin(phi)
            + (v2 * np.sin(h_ratio * (phi - phi12)))
            - (np.pi * 2 * radius
               * bending_rad * bdot * q_sign))


def rf_voltage_mch(phi: float, machine: 'Machine', rf_turn: int) -> float:
    """Objective function used in phi0 Newton optimization"""
    turn_time = machine.time_at_turn[rf_turn]
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    v2 = vrft(machine.vrf2, machine.vrf2dot, turn_time)

    return rf_voltage(phi, v1, v2, machine.bdot, machine.h_ratio,
                      machine.phi12, machine.mean_orbit_rad,
                      machine.bending_rad, machine.q)


def rf_voltage_pmch(phi: float, machine: 'ProgramsMachine', rf_turn: int) \
        -> float:
    """Objective function used in phi0 Newton optimization"""
    v1 = machine.vrf1_at_turn[rf_turn]
    v2 = machine.vrf2_at_turn[rf_turn]
    bdot = machine.bdot[rf_turn]
    phi12 = machine.phi12[rf_turn]

    return rf_voltage(phi, v1, v2, bdot, machine.h_ratio,
                      phi12, machine.mean_orbit_rad,
                      machine.bending_rad, machine.q)


def drf_voltage(phi: float, v1: arr_or_float, v2: arr_or_float,
                h_ratio: float, phi12: arr_or_float) -> arr_or_float:
    """Calculates derivative of RF voltage from both RF systems seen by
    particle.
    Needed by the Newton root finder to calculate phi0.

    Parameters
    ----------
    phi: float
        Phase where voltage is to be calculated.
    v1: np.ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.
    v2: np.ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.
    h_ratio: float
        Ratio of harmonics between the two RF systems.
    phi12: float or np.ndarray
        Phase difference between the two RF systems, turn-by-turn, or
        considered constant.

    Returns
    -------
    Derivative of voltage: float
        Derivative of voltage from both RF systems, as seen by particle.
    """
    return v1 * np.cos(phi) + h_ratio * v2 * np.cos(h_ratio * (phi - phi12))


def drf_voltage_mch(phi: float, machine: 'Machine', rf_turn: int) -> float:
    """Objective function used in phi0 Newton optimization"""
    turn_time = machine.time_at_turn[rf_turn]
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    v2 = vrft(machine.vrf2, machine.vrf2dot, turn_time)
    return (v1 * np.cos(phi)
            + machine.h_ratio
            * v2
            * np.cos(machine.h_ratio
                     * (phi - machine.phi12)))


def drf_voltage_pmch(phi: float, machine: 'ProgramsMachine', rf_turn: int) \
        -> float:
    """Objective function used in phi0 Newton optimization"""
    v1 = machine.vrf1_at_turn[rf_turn]
    v2 = machine.vrf2_at_turn[rf_turn]
    phi12 = machine.phi12[rf_turn]
    return (v1 * np.cos(phi) + machine.h_ratio * v2 * np.cos(machine.h_ratio
            * (phi - phi12)))


@dispatch(Number, Number, Number, Number, Number, Number, Number,
          (Sequence, np.ndarray), Number)
def rf_voltage_at_phase(phi: float, vrf1: float, vrf1dot: float, vrf2: float,
                        vrf2dot: float, h_ratio: float, phi12: float,
                        time_at_turns: np.ndarray, rf_turn: int):
    """RF voltage formula without calculating the difference in E0."""
    turn_time = time_at_turns[rf_turn]
    v1 = vrft(vrf1, vrf1dot, turn_time)
    v2 = vrft(vrf2, vrf2dot, turn_time)
    voltage = rf_voltage_at_phase(phi, v1, v2, h_ratio, phi12)
    return voltage


@dispatch(Number, Number, Number, Number, Number)
def rf_voltage_at_phase(phi: float, vrf1: float, vrf2: float, h_ratio: float,
                        phi12: float):
    voltage = vrf1 * np.sin(phi) + vrf2 * np.sin(h_ratio * (phi - phi12))
    return voltage


def vrft(vrf: float, vrf_dot: float, turn_time: float) -> float:
    """Calculates the RF peak voltage at turn rf_turn
    assuming a linear voltage function.
    Time=0 at machine reference frame.

    Parameters
    ----------
    vrf: float
        Peak voltage.
    vrf_dot: float
        Time derivative of RF voltage.
    turn_time: float
        time at turn for which the RF voltage should be calculated.

    Returns
    -------
    RF voltage: float
        RF voltage at turn.
    """
    return vrf + vrf_dot * turn_time


def find_synch_phase_mch(machine: 'Machine', rf_turn: int,
                         phi_lower: float, phi_upper: float) -> float:
    """Uses the Newton-Raphson root finder to estimate
    the synchronous phase for a particle on the normal orbit.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.
    rf_turn: int
        At which RF turn the calculation should happen.
    phi_lower:
        Lower phase boundary
    phi_upper:
        Upper phase boundary

    Returns
    -------
    synchronous phase: float
        Synchronous phase given in gradients.
    """
    try:
        phi_start = optimize.newton(func=rfvolt_rf1_mch,
                                    x0=(phi_lower + phi_upper) / 2.0,
                                    fprime=drfvolt_rf1_mch,
                                    tol=0.0001,
                                    maxiter=100,
                                    args=(machine, rf_turn))
        synch_phase = optimize.newton(func=rf_voltage_mch,
                                      x0=phi_start,
                                      fprime=drf_voltage_mch,
                                      tol=0.0001,
                                      maxiter=100,
                                      args=(machine, rf_turn))
    except RuntimeError:
        raise ValueError('Could not fit synchronous phase for the supplied '
                         'parameters')

    return synch_phase


def find_phi_lower_upper(machine: 'MachineABC', rf_turn: int) \
        -> t.Tuple[float, float]:
    """Calculates lower and upper phase of RF voltage
    for use in estimation of phi0.

    Parameters
    ----------
    machine: MachineABC
        Stores machine parameters and reconstruction settings.
    rf_turn: int
        At which RF turn the calculation should happen.

    Returns
    -------
    phi lower: float
        Lower phase boundary [rad].
    phi upper: float
        Upper phase boundary [rad].

    """
    condition = (machine.q
                 * (machine.trans_gamma
                    - (machine.e0[rf_turn]
                       / float(machine.e_rest)))) > 0
    if condition:
        phi_lower = -np.pi
        phi_upper = np.pi
    else:
        phi_lower = 0.0
        phi_upper = 2 * np.pi
    return phi_lower, phi_upper


def phase_slip_factor(machine: 'MachineABC') -> np.ndarray:
    """Calculates phase slip factor at each turn.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.

    Returns
    -------
    phase slip factor: ndarray
        1D array of phase slip factors for each turn.
    """
    return (1.0 - machine.beta0 ** 2) - machine.trans_gamma ** (-2)


def find_dphase(machine: 'MachineABC') -> np.ndarray:
    """Calculates coefficient needed for drift calculation during tracking.
    The drift coefficient is calculated for each machine turn.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.

    Returns
    -------
    drift factor: ndarray
        1D array holding drift coefficient for each machine turn.
    """
    return (2 * np.pi * machine.h_num * machine.eta0
            / (machine.e0 * machine.beta0 ** 2))


def revolution_freq(machine: 'MachineABC') -> np.ndarray:
    """Calculate revolution frequency for each turn.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.

    Returns
    -------
    revolution frequency: ndarray
        1D array holding revolution frequency at each machine turn.
    """
    return machine.beta0 * constants.c / machine.mean_orbit_rad


def calc_self_field_coeffs(machine: 'MachineABC') -> np.ndarray:
    """Calculates self-field coefficient for each profile.
    Needed for calculation of self-fields.

    Parameters
    ----------
    machine: Machine
        Stores machine parameters and reconstruction settings.

    Returns
    -------
    Self-field coefficients: ndarray
        1D array holding a self-field coefficient for each time frame.
    """
    sfc = np.zeros(machine.nprofiles)
    for i in range(machine.nprofiles):
        this_turn = i * machine.dturns
        sfc[i] = ((constants.e / machine.omega_rev0[this_turn])
                  * ((1.0 / machine.beta0[this_turn]
                      - machine.beta0[this_turn])
                     * machine.g_coupling * np.pi * 2.0e-7 * constants.c
                     - machine.zwall_over_n)
                  / machine.dtbin ** 2)
    return sfc


def phase_low(phase: float, bunch_phaselength: float, vrf1: np.ndarray,
              vrf2: np.ndarray, phi0: np.ndarray, h_ratio: float,
              phi12: arr_or_float, rf_turn: int,
              ref_frame: int = 0) -> float:
    """Calculates potential energy at phase.
    Needed for estimation of x-coordinate of synchronous particle.

    Parameters
    ----------
    phase: float
        Phase [rad] where potential energy is calculated.
    bunch_phaselength: float
        Length of bunch, given as phase [rad].
    vrf1: np.ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.
    vrf2: np.ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.
    phi0: ndarray
        1D array holding the synchronous phase angle at the end of each turn.
    h_ratio: float
        Ratio of harmonics between the two RF systems.
    phi12: float or np.ndarray
        Phase difference between the two RF systems, turn-by-turn, or
        considered constant.
    rf_turn: int
        Which turn to do the calculation at.
    ref_frame: int
        Frame to which machine parameters are referenced.
    """

    phi12 = phi12[ref_frame] if isinstance(phi12, np.ndarray) \
        or isinstance(phi12, Sequence) else phi12

    v1 = vrf1[ref_frame] * (np.cos(phase + bunch_phaselength) - np.cos(phase))

    v2 = (vrf2[ref_frame]
          * (np.cos(h_ratio
                    * (phase + bunch_phaselength - phi12))
             - np.cos(h_ratio * (phase - phi12)))
          / h_ratio)

    v_synch_phase = (bunch_phaselength
                     * rf_voltage_at_phase(phi0[rf_turn], vrf1[rf_turn],
                                           vrf2[rf_turn], h_ratio, phi12))

    return v1 + v2 + v_synch_phase


def phase_low_mch(phase: float, machine: 'MachineABC',
                  bunch_phaselength: float, rf_turn: int) -> float:
    """Calculates potential energy at phase.
    Needed for estimation of x-coordinate of synchronous particle.

    Parameters
    ----------
    phase: float
        Phase [rad] where potential energy is calculated.
    machine: Machine
        Stores machine parameters and reconstruction settings.
    bunch_phaselength: float
        Length of bunch, given as phase [rad]
    rf_turn: int
        At which RF turn the calculation should happen.
    """

    return phase_low(phase, bunch_phaselength, machine.vrf1_at_turn,
                     machine.vrf2_at_turn, machine.phi0, machine.h_ratio,
                     machine.phi12, rf_turn, machine.machine_ref_frame)


def dphase_low_mch(phase: float, machine: 'MachineABC',
                   bunch_phaselength: float, rf_turn: int, *args) -> float:
    """Calculates derivative of the phase_low function.
    The function is needed for estimation of x-coordinate
    of synchronous particle.

    Parameters
    ----------
    phase: float
        Phase [rad] where potential energy is calculated.
    machine: Machine
        Stores machine parameters and reconstruction settings.
    bunch_phaselength: float
        Length of bunch, given as phase [rad]
    rf_turn: int
        Which turn to do the calculation at.
    args: None
        Not used, arg needed for implementation of Newton-Raphson root finder.
    """
    return dphase_low(phase, bunch_phaselength, machine.vrf1_at_turn,
                      machine.vrf2_at_turn, machine.h_ratio,
                      machine.phi12, rf_turn)


def dphase_low(phase: float, bunch_phaselength: float, vrf1: np.ndarray,
               vrf2: np.ndarray, h_ratio: float, phi12: arr_or_float,
               rf_turn: int) -> float:
    """Calculates derivative of the phase_low function.
    The function is needed for estimation of x-coordinate
    of synchronous particle.

    Parameters
    ----------
    phase: float
        Phase [rad] where potential energy is calculated.
    bunch_phaselength: float
        Length of bunch, given as phase [rad].
    vrf1: np.ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.
    vrf2: np.ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.
    h_ratio: float
        Ratio of harmonics between the two RF systems.
    phi12: float or np.ndarray
        Phase difference between the two RF systems, turn-by-turn, or
        considered constant.
    rf_turn: int
        Which turn to do the calculation at.
    """
    phi12 = phi12[rf_turn] if isinstance(phi12, np.ndarray) or \
        isinstance(phi12, Sequence) else phi12

    ret = (-1.0 * vrf2[rf_turn]
           * (np.sin(h_ratio
                     * (phase + bunch_phaselength - phi12))
              - np.sin(h_ratio * (phase - phi12)))
           - vrf1[rf_turn]
           * (np.sin(phase + bunch_phaselength) - np.sin(phase)))

    return ret
