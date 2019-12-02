import logging as log
import numpy as np
from scipy import optimize

from .utils import assertions as asrt
from .utils import exceptions as expt
from . import physics

class Machine:

    def __init__(self, xat0, nprofiles, min_dt, max_dt, zwall_over_n,
                 g_coupling, charge, rest_energy, transitional_gamma,
                 h_num, h_ratio, phi12, b0, bdot, bending_radius,
                 mean_orbit_radius, vrf1, vrf1dot, vrf2, vrf2dot,
                 dturns, pickup_sensitivity, dtbin, demax, nbins, 
                 snpt=1, niter=20, machine_ref_frame=0, beam_ref_frame=0,
                 filmstart=0, filmstop=1, filmstep=1, output_dir=None,
                 self_field_flag=False, full_pp_flag=False):

        # TODO: Take rfv info as a single input
        # TODO: Take b-field info as a single input

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
    # (fitted_xat0, lower foot tangent, upper foot tangent)  
    def load_fitted_xat0_ftn(self, fit_info):
        log.info('Saving fitted xat0 to machine object.')
        self.fitted_xat0 = fit_info[0]
        self.tangentfoot_low = fit_info[1]
        self.tangentfoot_up = fit_info[2]
        self.xat0 = self.fitted_xat0

    # Calculating values that changes for each m. turn.
    # First is the arrays inited at index of machine ref. frame (i0).
    # Based on this value are the rest of the values calculated;
    # first, upwards from i0 to total number of turns + 1, then downwards from i0 to 0 (first turn).
    def values_at_turns(self):
        # Add input assertions.
        all_turns = (self.nprofiles - 1) * self.dturns
        self._init_arrays(all_turns)
        i0 = self._array_initial_values()

        for i in range(i0 + 1, all_turns + 1):
            self.time_at_turn[i] = (self.time_at_turn[i - 1]
                                    + 2 * np.pi * self.mean_orbit_rad
                                    / (self.beta0[i - 1] * physics.C))
         
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
                                    / (self.beta0[i] * physics.C))
            
            self.phi0[i] = optimize.newton(func=physics.rf_voltage,
                                           x0=self.phi0[i + 1],
                                           fprime=physics.drf_voltage,
                                           tol=0.0001,
                                           maxiter=100,
                                           args=(self, i))

        # Calculate phase slip factor at each turn
        self.eta0 = physics.phase_slip_factor(self)

        # Calculates dphase for each turn
        self.dphase = physics.find_dphase(self)

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
        self.dphase = np.zeros(array_length)
        self.deltaE0 = np.zeros(array_length)
        self.beta0 = np.zeros(array_length)
        self.eta0 = np.zeros(array_length)
        self.e0 = np.zeros(array_length)

    # Calculating start-values for the parameters that changes for each turn.
    # The reference frame where the start-values are calculated is the machine reference frame.
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
