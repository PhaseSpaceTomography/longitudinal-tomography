# Settings for reconstruction:
# ----------------------------------------------------------------------------
# xat0              Synchronous phase in bins in beam_ref_frame
#                   read form file as time (in frame bins) from the lower
#                   profile bound to the synchronous phase
#                   (if < 0, a fit is performed) in the bunch ref. frame
# yat0              Synchronous energy (0 in relative terms)
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

from .utils import assertions as asrt
from .utils import exceptions as expt
from . import physics

class Machine:

    def __init__(self, xat0, nprofiles, zwall_over_n, g_coupling,
                 charge, rest_energy, transitional_gamma,
                 h_num, h_ratio, phi12, b0, bdot, bending_radius,
                 mean_orbit_radius, vrf1, vrf1dot, vrf2, vrf2dot,
                 dturns, pickup_sensitivity, dtbin, demax, nbins,
                 min_dt=None, max_dt=None, snpt=1, niter=20,
                 machine_ref_frame=0, beam_ref_frame=0,
                 filmstart=0, filmstop=1, filmstep=1, output_dir=None,
                 self_field_flag=False, full_pp_flag=False):

        # TODO: Take rfv info as a single input
        # TODO: Take b-field info as a single input

        if min_dt == None:
            min_dt = 0.0
        if max_dt == None:
            max_dt = nbins * dtbin

        # Machine parameters
        self.demax = demax
        self.dturns = dturns
        self.vrf1 = vrf1
        self.vrf2 = vrf2
        self.vrf1dot = vrf1dot
        self.vrf2dot = vrf2dot
        self.mean_orbit_rad = mean_orbit_radius
        self.bending_rad = bending_radius
        self.b0 = b0
        self.bdot = bdot
        self.phi12 = phi12
        self.h_ratio = h_ratio
        self.h_num = h_num
        self.trans_gamma = transitional_gamma
        self.e_rest = rest_energy
        self.q = charge
        self.g_coupling = g_coupling
        self.zwall_over_n = zwall_over_n
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.nprofiles = nprofiles
        self.pickup_sensitivity = pickup_sensitivity
        self.xat0 = xat0
        self.dtbin = dtbin
        self.nbins = nbins

        # Flags
        self.self_field_flag = self_field_flag
        self.full_pp_flag = full_pp_flag

        # Reconstruction parameters
        self.machine_ref_frame = machine_ref_frame
        self.beam_ref_frame = beam_ref_frame
        self.snpt = snpt
        self.niter = niter
        self.filmstart = filmstart
        self.filmstop = filmstop
        self.filmstep = filmstep
        self.output_dir = output_dir

    @property
    def nbins(self):
        return self._nbins

    @nbins.setter
    def nbins(self, in_nbins):
        self._nbins = in_nbins
        self._find_yat0()
        log.info(f'yat0 was updated when the '
                 f'number of profile bins changed.\nNew values - '
                 f'nbins: {self.nbins}, yat0: {self.yat0}')

    # Function for setting the xat0 if a fit has been performed.
    # Saves parameters gathered from fit, needed by the 'print_plotinfo'
    #  function.
    # Fit info should be a tuple of the following format:
    # (fitted_xat0, lower fbunch limit, upper bunch limit)
    # fitted_xat0 must be saved for fortran output: 'print_plotinfo'  
    def load_fitted_xat0_ftn(self, fit_info):
        log.info('Saving fitted xat0 to machine object.')
        self.fitted_xat0 = fit_info[0]
        self.bunchlimit_low = fit_info[1]
        self.bunchlimit_up = fit_info[2]
        self.xat0 = self.fitted_xat0

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

    def _find_yat0(self):
        self.yat0 = self.nbins / 2.0

    def rfv_at_turns(self):
        rf1v = self.vrf1 + self.vrf1dot * self.time_at_turn
        rf2v = self.vrf2 + self.vrf2dot * self.time_at_turn
        return rf1v, rf2v
