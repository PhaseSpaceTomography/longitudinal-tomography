# Settings for reconstruction:
# ----------------------------------------------------------------------------
# synch_part_x      Synchronous phase in bins in beam_ref_frame
#                   read form file as time (in frame bins) from the lower
#                   profile bound to the synchronous phase
#                   (if < 0, a fit is performed) in the bunch ref. frame
# synch_part_y      Synchronous energy (0 in relative terms)
#                   in reconstructed phase space coordinate system
# dtbin             bin with [s] 
# demax             maximum energy of reconstructed phase space
# max_dt            Minimum phase of reconstructed phase space,
#                   smaler phases are threated as empty.
# max_dt            Maximum phase of reconstructed phase space,
#                   larger phases are threated as empty. 
# snpt              Square root of number of test particles tracked from each
#                   pixel of reconstructed phase space
# niter             Number of iterations in the reconstruction process
# machine_ref_frame Frame to which machine parameters are referenced
# beam_ref_frame    Frame to which beam parameters are referenced
# filmstart         First profile to be reconstructed
# filmstop          Last profile to be reconstructed
# filmstep          Step between consecutive reconstructions
#                   for the profiles from filmstart to filmstop
# full_pp_flag      If set, all pixels in reconstructed
#                   phase space will be tracked
#
# Machine and Particle Parameters:
# ----------------------------------------------------------------------------
# vrf1, vrf2        Peak voltage of first and second RF
#                   system at machine_ref_frame
# vrfXdot           Time derivatives of the RF voltages (considered constant)
# mean_orbit_rad    Machine mean orbit radius       [m]
# bending_rad       Machine bending radius          [m]
# b0                B-field at machine_ref_frame    [T]
# bdot              Time derivative of B-field (considered constant) [T/s]
# phi12             Phase difference between the two RF systems
#                   (considered constant)
# h_ratio           Ratio of harmonics between the two RF systems
# h_num             Principle harmonic number
# trans_gamma       Transitional gamma
# e_rest            Rest energy of accelerated particle     [eV/C^2]
# q                 Charge state of accelerated particle
#
# Space charge parameters:
# ----------------------------------------------------------------------------
# self_field_flag       Flag to include self-fields in the tracking.
# g_coupling            Space charge coupling coefficient
#                       (geometrical coupling coefficient)
# zwall_over_n          Magnitude of Zwall/n, reactive impedance
#                       (in Ohms per mode number) over a machine turn
# pickup_sensitivity    Effective pick-up sensitivity
#                       (in digitizer units per instantaneous Amp)
#
# Calculated arrays:
#-----------------------------------------------------------------------------
# time_at_turn      Time at each turn,
#                   relative to machine_ref_frame at the end of each turn.
# omega_rev0        Revolution frequency at each turn.
# phi0              Synchronous phase angle at the end of each turn.
# drift_coef        Coefficient used for calculating difference,
#                   from phase n to phase n + 1.
#                   Needed in trajectory height calculator and tracking.
# beta0             Lorenz beta factor (v/c) at the end of each turn
# eta0              Phase slip factor at each turn
# e0                Total energy of synchronous particle
#                   at the end of each turn.
# deltaE0           Difference between e0(n) and e0(n-1) for each turn.
#


import logging as log
import numpy as np
from scipy import optimize, constants

from ..utils import assertions as asrt
from ..utils import exceptions as expt
from ..utils import physics

_machine_opts_def = {}
_machine_opts_def['demax'] = -1.E6
_machine_opts_def['vrf1dot'] = 0.0 
_machine_opts_def['vrf2'] = 0.0 
_machine_opts_def['vrf2dot'] = 0.0 
_machine_opts_def['phi12'] = 0.0
_machine_opts_def['h_ratio'] = 1.0
_machine_opts_def['h_num'] = 1 
_machine_opts_def['charge'] = 1
_machine_opts_def['g_coupling'] = None
_machine_opts_def['zwall_over_n'] = None
_machine_opts_def['min_dt'] = None 
_machine_opts_def['max_dt'] = None
_machine_opts_def['self_field_flag'] = False 
_machine_opts_def['full_pp_flag'] = False
_machine_opts_def['pickup_sensitivity'] = None 
_machine_opts_def['machine_ref_frame'] = 0 
_machine_opts_def['beam_ref_frame'] = 0 
_machine_opts_def['snpt'] = 4
_machine_opts_def['niter'] = 20
_machine_opts_def['filmstart'] = 0
_machine_opts_def['filmstop'] = 1
_machine_opts_def['filmstep'] = 1
_machine_opts_def['output_dir'] = None

default_opts = {}
def _reset_defaults():
    default_opts.update({key: _machine_opts_def[key]
                        for key in _machine_opts_def})
    for item in tuple(default_opts.keys()):
        if item not in _machine_opts_def:
            default_opts.pop(item)
_reset_defaults()

def _assert_machine_kwargs(**kwargs):
    use_params = {}

    for item in default_opts:
        use_params[item] = default_opts[item]

    for item in kwargs:
        if item not in default_opts:
            raise KeyError(f'{item} is not a machine parameter')
        else:
            use_params[item] = kwargs[item]
    return use_params

class Machine:

    def __init__(self, dturns, vrf1, mean_orbit_rad, bending_rad,
                 b0, bdot, trans_gamma, rest_energy, nprofiles, nbins,
                 synch_part_x, dtbin, **kwargs):

        kwargs = _assert_machine_kwargs(**kwargs)

        # TODO: Take rfv info as a single input
        # TODO: Take b-field info as a single input

        if kwargs['min_dt'] == None:
            min_dt = 0.0
        else:
            min_dt = kwargs['min_dt']

        if kwargs['max_dt'] == None:
            max_dt = nbins * dtbin
        else:
            max_dt = kwargs['max_dt']

        # Machine parameters
        self.demax = kwargs['demax']
        self.dturns = dturns
        self.vrf1 = vrf1
        self.vrf1dot = kwargs['vrf1dot']
        self.vrf2 = kwargs['vrf2']
        self.vrf2dot = kwargs['vrf2dot']
        self.mean_orbit_rad = mean_orbit_rad
        self.bending_rad = bending_rad
        self.b0 = b0
        self.bdot = bdot
        self.phi12 = kwargs['phi12']
        self.h_ratio = kwargs['h_ratio']
        self.h_num = kwargs['h_num']
        self.trans_gamma = trans_gamma
        self.e_rest = rest_energy
        self.q = kwargs['charge']
        self.g_coupling = kwargs['g_coupling']
        self.zwall_over_n = kwargs['zwall_over_n']
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.nprofiles = nprofiles
        self.pickup_sensitivity = kwargs['pickup_sensitivity']
        self.nbins = nbins
        self.synch_part_x = synch_part_x
        self.dtbin = dtbin

        # Flags
        self.self_field_flag = kwargs['self_field_flag']
        self.full_pp_flag = kwargs['full_pp_flag']

        # Reconstruction parameters
        self.machine_ref_frame = kwargs['machine_ref_frame']
        self.beam_ref_frame = kwargs['beam_ref_frame']
        self.snpt = kwargs['snpt']
        self.niter = kwargs['niter']
        self.filmstart = kwargs['filmstart']
        self.filmstop = kwargs['filmstop']
        self.filmstep = kwargs['filmstep']
        self.output_dir = kwargs['output_dir']

    @property
    def nbins(self):
        return self._nbins

    @nbins.setter
    def nbins(self, in_nbins):
        self._nbins = in_nbins
        self._find_synch_part_y()
        log.info(f'synch_part_y was updated when the '
                 f'number of profile bins changed.\nNew values - '
                 f'nbins: {self.nbins}, synch_part_y: {self.synch_part_y}')

    # Function for setting the synch_part_x if a fit has been performed.
    # Saves parameters gathered from fit, needed by the 'print_plotinfo'
    #  function.
    # Fit info should be a tuple of the following format:
    # (fitted_synch_part_x, lower fbunch limit, upper bunch limit)
    # fitted_synch_part_x must be saved for fortran output: 'print_plotinfo'  
    def load_fitted_synch_part_x_ftn(self, fit_info):
        log.info('Saving fitted synch_part_x to machine object.')
        self.fitted_synch_part_x = fit_info[0]
        self.bunchlimit_low = fit_info[1]
        self.bunchlimit_up = fit_info[2]
        self.synch_part_x = self.fitted_synch_part_x

    # Calculating values that changes for each m. turn.
    # First is the arrays inited at index of machine ref. frame (i0).
    # Based on this value are the rest of the values calculated;
    # first, upwards from i0 to total number of turns + 1,
    # then downwards from i0 to 0 (first turn).
    def values_at_turns(self):
        # Add input assertions.
        asrt.assert_machine_input(self)
        all_turns = (self.nprofiles - 1) * self.dturns
        self._init_arrays(all_turns)
        i0 = self._array_initial_values()

        for i in range(i0 + 1, all_turns + 1):
            self.time_at_turn[i] = (self.time_at_turn[i - 1]
                                    + 2 * np.pi * self.mean_orbit_rad
                                    / (self.beta0[i - 1] * constants.c))
         
            self.phi0[i] = optimize.newton(func=physics.rf_voltage,
                                           x0=self.phi0[i - 1],
                                           fprime=physics.drf_voltage,
                                           tol=0.0001,
                                           maxiter=100,
                                           args=(self, i))
            
            self.e0[i] = (self.e0[i - 1]
                          + self.q
                          * physics.short_rf_voltage_formula(
                                self.phi0[i], self.vrf1, self.vrf1dot,
                                self.vrf2, self.vrf2dot, self.h_ratio,
                                self.phi12, self.time_at_turn, i))

            self.beta0[i] = np.sqrt(1.0 - (self.e_rest/float(self.e0[i]))**2)
            self.deltaE0[i] = self.e0[i] - self.e0[i - 1]
        for i in range(i0 - 1, -1, -1):
            self.e0[i] = (self.e0[i + 1]
                          - self.q
                          * physics.short_rf_voltage_formula(
                                self.phi0[i + 1], self.vrf1, self.vrf1dot,
                                self.vrf2, self.vrf2dot, self.h_ratio,
                                self.phi12, self.time_at_turn, i + 1))

            self.beta0[i] = np.sqrt(1.0 - (self.e_rest/self.e0[i])**2)
            self.deltaE0[i] = self.e0[i + 1] - self.e0[i]

            self.time_at_turn[i] = (self.time_at_turn[i + 1]
                                    - 2 * np.pi * self.mean_orbit_rad
                                    / (self.beta0[i] * constants.c))
            
            self.phi0[i] = optimize.newton(func=physics.rf_voltage,
                                           x0=self.phi0[i + 1],
                                           fprime=physics.drf_voltage,
                                           tol=0.0001,
                                           maxiter=100,
                                           args=(self, i))

        # Calculate phase slip factor at each turn
        self.eta0 = physics.phase_slip_factor(self)

        # Calculates dphase for each turn
        self.drift_coef = physics.find_dphase(self)

        # Calculate revolution frequency at each turn
        self.omega_rev0 = physics.revolution_freq(self)

        # Calculate RF-voltages at each turn
        self.vrf1_at_turn, self.vrf2_at_turn = self.rfv_at_turns()

    # Initiating arrays in order to store information about parameters
    # that has a different value every turn.
    def _init_arrays(self, all_turns):
        array_length = all_turns + 1
        self.time_at_turn = np.zeros(array_length)
        self.omega_rev0 = np.zeros(array_length)
        self.phi0 = np.zeros(array_length)
        self.drift_coef = np.zeros(array_length)
        self.deltaE0 = np.zeros(array_length)
        self.beta0 = np.zeros(array_length)
        self.eta0 = np.zeros(array_length)
        self.e0 = np.zeros(array_length)

    # Calculating start-values for the parameters that changes for each turn.
    # The reference frame where the start-values
    # are calculated is the machine reference frame.
    # (machine ref. frame -1 to adjust for fortran input files)
    def _array_initial_values(self):
        i0 = self.machine_ref_frame * self.dturns
        self.time_at_turn[i0] = 0
        self.e0[i0] = physics.b_to_e(self)
        self.beta0[i0] = physics.lorenz_beta(self, i0)
        phi_lower, phi_upper = physics.find_phi_lower_upper(self, i0)
        # Synchronous phase of a particle on the nominal orbit
        self.phi0[i0] = physics.find_synch_phase(self, i0, phi_lower,
                                                 phi_upper)
        return i0

    def _find_synch_part_y(self):
        self.synch_part_y = self.nbins / 2.0

    def rfv_at_turns(self):
        rf1v = self.vrf1 + self.vrf1dot * self.time_at_turn
        rf2v = self.vrf2 + self.vrf2dot * self.time_at_turn
        return rf1v, rf2v
