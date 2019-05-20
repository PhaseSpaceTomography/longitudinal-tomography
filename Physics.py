"""
    Physics formulas
                    """
from Numeric import *
from numba import njit

# Constants:
C = 2.99792458e8
e_UNIT = 1.60217733e-19


# Calculates the energy for a particle in
# a circular machine at dipole field B.
def b_to_e(parameters):
    return np.sqrt((parameters.q
                    * parameters.b0
                    * parameters.bending_rad
                    * C)**2
                   + parameters.e_rest**2)


# Calculates Lorenz beta factor (v/c) at a turn
def lorenz_beta(parameters, rf_turn):
    return np.sqrt(1.0 - (float(parameters.e_rest) /
                          parameters.e0[rf_turn])**2)


# Needed by the Newton root finder to calculate phi0
def rfvolt_rf1(phi, parameters, rf_turn):
    turn_time = parameters.time_at_turn[rf_turn]
    v1 = vrft(parameters.vrf1, parameters.vrf1dot, turn_time)
    q_sign = np.sign([parameters.q])
    return (v1 * np.sin(phi)
            - 2 * np.pi * parameters.mean_orbit_rad
                * parameters.bending_rad * parameters.bdot * q_sign)


# Needed by the Newton root finder to calculate phi0
def drfvolt_rf1(phi, parameters, rf_turn):
    turn_time = parameters.time_at_turn[rf_turn]
    v1 = vrft(parameters.vrf1, parameters.vrf1dot, turn_time)
    return v1 * np.cos(phi)


# Needed by the Newton root finder to calculate phi0
def rf_voltage(phi, parameters, rf_turn):
    turn_time = parameters.time_at_turn[rf_turn]
    q_sign = np.sign([parameters.q])
    v1 = vrft(parameters.vrf1, parameters.vrf1dot, turn_time)
    v2 = vrft(parameters.vrf2, parameters.vrf2dot, turn_time)
    return (v1 * np.sin(phi)
            + (v2 * np.sin(parameters.h_ratio * (phi - parameters.phi12)))
            - (np.pi * 2 * parameters.mean_orbit_rad
               * parameters.bending_rad * parameters.bdot * q_sign[0]))


# Needed by the Newton root finder to calculate phi0
def drf_voltage(phi, parameters, rf_turn):
    turn_time = parameters.time_at_turn[rf_turn]
    v1 = vrft(parameters.vrf1, parameters.vrf1dot, turn_time)
    v2 = vrft(parameters.vrf2, parameters.vrf2dot, turn_time)
    return (v1 * np.cos(phi)
            + parameters.h_ratio
            * v2
            * np.cos(parameters.h_ratio
                     * (phi - parameters.phi12)))

def short_rf_voltage_formula(phi, vrf1, vrf1dot, vrf2, vrf2dot,
                             h_ratio, phi12, time_at_turn, rf_turn):
    turn_time = time_at_turn[rf_turn]
    v1 = vrft(vrf1, vrf1dot, turn_time)
    v2 = vrft(vrf2, vrf2dot, turn_time)
    temp = v1 * np.sin(phi)
    temp += v2 * np.sin(h_ratio * (phi - phi12))
    return temp

# Calculates the RF peak voltage at turn rf_turn
#   assuming a linear voltage function vrft=vrfdot*time+vrf.
#   time=0 at machine_ref_frame.
@njit
def vrft(vrf, vrfDot, turn_time):
    #   time_at_turn: time at turn for which the RF voltage should be calculated
    #   vrf: Volts
    #   vrfDto: Volts/s
    return vrf + vrfDot * turn_time


# Synchronous phase for a particle on the normal orbit
def find_synch_phase(parameters, rf_turn, phi_lower, phi_upper):
    phi_start = newton(rfvolt_rf1, drfvolt_rf1,
                       (phi_lower + phi_upper) / 2.0,
                       parameters, rf_turn, 0.001)
    return newton(rf_voltage, drf_voltage,
                  phi_start, parameters, rf_turn, 0.001)


def find_phi_lower_upper(parameters, rf_turn):
    condition = (parameters.q
                 * (parameters.trans_gamma
                    - (parameters.e0[rf_turn]
                        / float(parameters.e_rest)))) > 0
    if condition:
        phi_lower = -1.0 * np.pi
        phi_upper = np.pi
    else:
        phi_lower = 0.0
        phi_upper = 2 * np.pi
    return phi_lower, phi_upper


# Finds phase slip factor at each turn.
def phase_slip_factor(parameters):
    return (1.0 - parameters.beta0**2) - parameters.trans_gamma**(-2)


# Find coefficient "c1" at each turn, TODO: find out what it is.
def find_c1(parameters):
    return (2 * np.pi * parameters.h_num * parameters.eta0
            / (parameters.e0 * parameters.beta0**2))


# Find revolution frequency at each turn
def revolution_freq(parameters):
    return parameters.beta0 * C / parameters.mean_orbit_rad


# Calculates self field coefficient for each profile
def calc_self_field_coeffs(parameters):
    sfc = np.zeros(parameters.profile_count)
    for i in range(parameters.profile_count):
        this_turn = i * parameters.dturns
        sfc[i] = ((e_UNIT / parameters.omega_rev0[this_turn])
                  * ((1.0 / parameters.beta0[this_turn]
                     - parameters.beta0[this_turn])
                     * parameters.g_coupling * np.pi * 2.0e-7 * C
                     - parameters.zwall_over_n)
                  / parameters.dtbin**2)
    return sfc


# Calculates potential energy in phase? TODO: Find better explanation
def phase_low(phase, parameters, rf_turn):
    term1 = (parameters.vrf2
             * (np.cos(parameters.h_ratio
                       * (phase
                          + parameters.bunch_phaselength
                          - parameters.phi12))
                - np.cos(parameters.h_ratio
                         * (phase - parameters.phi12)))
             / parameters.h_ratio)
    term2 = (parameters.vrf1
             * (np.cos(phase + parameters.bunch_phaselength)
                - np.cos(phase)))
    term3 = (parameters.bunch_phaselength
             * short_rf_voltage_formula(parameters.phi0[rf_turn],
                                        parameters.vrf1,
                                        parameters.vrf1dot,
                                        parameters.vrf2,
                                        parameters.vrf2dot,
                                        parameters.h_ratio,
                                        parameters.phi12,
                                        parameters.time_at_turn,
                                        rf_turn))
    return term1 + term2 + term3


# Calculates derivative of phase_low() TODO: Find better expl.
# Third argument is needed for newton func.
def dphase_low(phase, parameters, rf_turn):
    return (-1.0 * parameters.vrf2
            * (np.sin(parameters.h_ratio
                      * (phase + parameters.bunch_phaselength - parameters.phi12))
                - np.sin(parameters.h_ratio * (phase - parameters.phi12)))
            - parameters.vrf1
            * (np.sin(phase + parameters.bunch_phaselength) - np.sin(phase)))




