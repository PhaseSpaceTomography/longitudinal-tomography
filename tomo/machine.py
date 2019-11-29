import logging as log
import numpy as np
from scipy import optimize

from . import physics
from .utils import assertions as asrt
from .utils import exceptions as expt

class Machine:

    def __init__(self):

        self.output_dir = None
        self._xat0 = None
        self._dtbin = None
        self.demax = None
        self.dturns = None
        self.snpt = None
        self.niter = None
        self.machine_ref_frame = None
        self.beam_ref_frame = None
        self.filmstart = None
        self.filmstop = None
        self.filmstep = None
        self.full_pp_flag = None
        self.vrf1 = None
        self.vrf2 = None
        self.vrf1dot = None
        self.vrf2dot = None
        self.mean_orbit_rad = None
        self.bending_rad = None
        self.b0 = None
        self.bdot = None
        self.phi12 = None
        self.h_ratio = None
        self.h_num = None
        self.trans_gamma = None
        self.e_rest = None
        self.q = None
        self.self_field_flag = None
        self.g_coupling = None
        self.zwall_over_n = None
        self.pickup_sensitivity = None
        self.max_dt = None
        self.min_dt = None
        self.nprofiles = None

    @property
    def xat0(self):
        return self._xat0

    @xat0.setter
    def xat0(self, in_xat0):
        self._xat0 = in_xat0
        self._calc_xorigin()
        log.info(f'xorigin was updated when '\
                 f'the value of xat0 was changed.\nNew values - '
                 f'xat0: {self.xat0}, xorigin: {self.xorigin}')

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

    @property
    def dtbin(self):
        return self._dtbin

    @dtbin.setter
    def dtbin(self, value):
        self._dtbin = value
        self._calc_xorigin()
        log.info(f'xorigin was updated when '\
                 f'the value of dtbin was changed.\nNew values - '
                 f'dtbin: {self.dtbin}, xorigin: {self.xorigin}')

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
        all_turns = self._calc_number_of_turns()
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

    # Calculate the absolute difference (in bins) between phase=0 and
    # origin of the reconstructed phase space coordinate system.
    def _calc_xorigin(self):
        reference_turn = self.beam_ref_frame * self.dturns
        self.xorigin = (self.phi0[reference_turn]
                        / (self.h_num
                           * self.omega_rev0[reference_turn]
                           * self.dtbin)
                        - self.xat0)

    def _find_yat0(self):
        self.yat0 = self.nbins / 2.0

    def rfv_at_turns(self):
        rf1v = self.vrf1 + self.vrf1dot * self.time_at_turn
        rf2v = self.vrf2 + self.vrf2dot * self.time_at_turn
        return rf1v, rf2v

    # Calculating total number of machine turns
    def _calc_number_of_turns(self):
        all_turns = (self.nprofiles - 1) * self.dturns
        asrt.assert_greater(all_turns, 'all_turns', 0, expt.InputError,
                            'Make sure that frame skip-count'
                            'do not exceed number of frames')
        return all_turns
