import numpy as np
from scipy import optimize, constants


"""
    Physics formulas
                    """

# Calculates the energy for a particle in
# a circular machine at dipole field B.
def b_to_e(machine):
    return np.sqrt((machine.q
                    * machine.b0
                    * machine.bending_rad
                    * constants.c)**2
                   + machine.e_rest**2)


# Calculates Lorenz beta factor (v/c) at a turn
def lorenz_beta(machine, rf_turn):
    return np.sqrt(1.0 - (float(machine.e_rest) /
                          machine.e0[rf_turn])**2)


# Needed by the Newton root finder to calculate phi0
def rfvolt_rf1(phi, machine, rf_turn):
    turn_time = machine.time_at_turn[rf_turn]
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    q_sign = np.sign([machine.q])
    return (v1 * np.sin(phi)
            - 2 * np.pi * machine.mean_orbit_rad
                * machine.bending_rad * machine.bdot * q_sign)

# Needed by the Newton root finder to calculate phi0
def drfvolt_rf1(phi, machine, rf_turn):
    turn_time = machine.time_at_turn[rf_turn]
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    return v1 * np.cos(phi)


# Needed by the Newton root finder to calculate phi0
def rf_voltage(phi, machine, rf_turn):
    turn_time = machine.time_at_turn[rf_turn]
    q_sign = np.sign([machine.q])
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    v2 = vrft(machine.vrf2, machine.vrf2dot, turn_time)
    return (v1 * np.sin(phi)
            + (v2 * np.sin(machine.h_ratio * (phi - machine.phi12)))
            - (np.pi * 2 * machine.mean_orbit_rad
               * machine.bending_rad * machine.bdot * q_sign[0]))


# Needed by the Newton root finder to calculate phi0
def drf_voltage(phi, machine, rf_turn):
    turn_time = machine.time_at_turn[rf_turn]
    v1 = vrft(machine.vrf1, machine.vrf1dot, turn_time)
    v2 = vrft(machine.vrf2, machine.vrf2dot, turn_time)
    return (v1 * np.cos(phi)
            + machine.h_ratio
            * v2
            * np.cos(machine.h_ratio
                     * (phi - machine.phi12)))


# RF voltage formula without calculating the difference in E0
def short_rf_voltage_formula(phi, vrf1, vrf1dot, vrf2, vrf2dot,
                             h_ratio, phi12, time_at_turn, rf_turn):
    turn_time = time_at_turn[rf_turn]
    v1 = vrft(vrf1, vrf1dot, turn_time)
    v2 = vrft(vrf2, vrf2dot, turn_time)
    temp = v1 * np.sin(phi)
    temp += v2 * np.sin(h_ratio * (phi - phi12))
    return temp


# Calculates the RF peak voltage at turn rf_turn
# assuming a linear voltage function
# time=0 at machine_ref_frame.
def vrft(vrf, vrfDot, turn_time):
    #   time_at_turn: time at turn for which the RF voltage should be calculated
    #   vrf: Volts
    #   vrfDto: Volts/s
    return vrf + vrfDot * turn_time


# Synchronous phase for a particle on the normal orbit
def find_synch_phase(machine, rf_turn, phi_lower, phi_upper):
    phi_start = optimize.newton(func=rfvolt_rf1,
                                x0=(phi_lower + phi_upper) / 2.0,
                                fprime=drfvolt_rf1,
                                tol=0.0001,
                                maxiter=100,
                                args=(machine, rf_turn))
    synch_phase = optimize.newton(func=rf_voltage,
                                  x0=phi_start,
                                  fprime=drf_voltage,
                                  tol=0.0001,
                                  maxiter=100,
                                  args=(machine, rf_turn))
    return synch_phase


def find_phi_lower_upper(machine, rf_turn):
    condition = (machine.q
                 * (machine.trans_gamma
                    - (machine.e0[rf_turn]
                        / float(machine.e_rest)))) > 0
    if condition:
        phi_lower = -1.0 * np.pi
        phi_upper = np.pi
    else:
        phi_lower = 0.0
        phi_upper = 2 * np.pi
    return phi_lower, phi_upper


# Finds phase slip factor at each turn.
def phase_slip_factor(machine):
    return (1.0 - machine.beta0**2) - machine.trans_gamma**(-2)


# Find dphase at each turn
def find_dphase(machine):
    return (2 * np.pi * machine.h_num * machine.eta0
            / (machine.e0 * machine.beta0**2))


# Find revolution frequency at each turn
def revolution_freq(machine):
    return machine.beta0 * constants.c / machine.mean_orbit_rad


# Calculates self field coefficient for each profile
def calc_self_field_coeffs(machine):
    sfc = np.zeros(machine.nprofiles)
    for i in range(machine.nprofiles):
        this_turn = i * machine.dturns
        sfc[i] = ((constants.e / machine.omega_rev0[this_turn])
                  * ((1.0 / machine.beta0[this_turn]
                     - machine.beta0[this_turn])
                     * machine.g_coupling * np.pi * 2.0e-7 * constants.c
                     - machine.zwall_over_n)
                  / machine.dtbin**2)
    return sfc


# Calculates potential energy in phase
def phase_low(phase, machine, bunch_phaselength, rf_turn):
    term1 = (machine.vrf2
             * (np.cos(machine.h_ratio
                       * (phase
                          + bunch_phaselength
                          - machine.phi12))
                - np.cos(machine.h_ratio
                         * (phase - machine.phi12)))
             / machine.h_ratio)
    term2 = (machine.vrf1
             * (np.cos(phase + bunch_phaselength)
                - np.cos(phase)))
    term3 = (bunch_phaselength
             * short_rf_voltage_formula(
                machine.phi0[rf_turn], machine.vrf1, machine.vrf1dot,
                machine.vrf2, machine.vrf2dot, machine.h_ratio,
                machine.phi12, machine.time_at_turn, rf_turn))
    return term1 + term2 + term3


# Calculates derivative of the phase_low function.
# *args is needed for Newton-Raphson root finder.
def dphase_low(phase, machine, bunch_phaselength, *args):
    return (-1.0 * machine.vrf2
            * (np.sin(machine.h_ratio
                      * (phase + bunch_phaselength - machine.phi12))
                - np.sin(machine.h_ratio * (phase - machine.phi12)))
            - machine.vrf1
            * (np.sin(phase + bunch_phaselength) - np.sin(phase)))
