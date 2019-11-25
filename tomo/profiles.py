import logging as log
from scipy import signal
from scipy import optimize
import numpy as np
from machine import Machine
from physics import phase_low, dphase_low, e_UNIT
from utils.assertions import assert_equal, assert_inrange, assert_greater
from utils.exceptions import (RawDataImportError, InputError,
                               RebinningError)
# ================
# About TimeSpace:   <- fix!
# ================
# The TimeSpace class handles import and processing of data in time domain.
#  - importing raw data, and converting it to profiles, specified by input parameters.
#  - calculation of total charge in profile
#  - if xat0, from input parameters, is less than 0 wil a fit be performed to find xat0
#  - calculations of yat0, phi wrap, xorigin
#  - if self field flag is true will self fields be included in the reconstruction.
#       in the time space class will a savitzky-golay smoothing filter be applied to the profiles
#       and the self field voltage will be calculated.
#
# Variables used in this class are retrieved from a parameter object,
#   which stores the input parameters for the reconstruction.
#   A description of all the input and time space variables can be found in the parameters module.
#
# ====================
# TimeSpace variables:
# ====================
# par               parameter object (for more info, see parameters module)
# profiles          a (profile-count, profile-length) shaped matrix containing profile data, ready for tomography.
# profile_charge    Total charge in the reference profile
#
# Self field variables:
# ---------------------
# vself             Self-field voltage
# dsprofiles        Smoothed derivative of profiles
#
# bunch_phaselength Bunch phase length in beam reference profile
# tangentfoot_low   Used for estimation of bunch duration
# tangentfoot_up
# phiwrap           Phase covering the an integer of rf periods
# wrap_length       Maximum number of bins to cover an integer number of rf periods
# fitted_xat0          Value of (if) fitted xat0


class Profiles:

    def __init__(self, machine):
        self.machine = machine
        self._waterfall = None
        self.profile_charge = None   # Total charge in reference profile


        # Self field variables:
        # self.vself = None            # Self-field voltage
        # self.dsprofiles = None       # Smoothed derivative of profiles

        # To be moved out of class?
        self.tangentfoot_low = 0.0
        self.tangentfoot_up = 0.0
        self.phiwrap = 0.0
        self.wrap_length = 0
        self.fitted_xat0 = 0.0

    # ================ NEW ================

    @property
    def waterfall(self):
        return self._waterfall

    @waterfall.setter
    def waterfall(self, new_waterfall):
        if not hasattr(new_waterfall, '__iter__'):
            raise InputError('waterfall should be an iterable.')
        log.info('Loading waterfall...')
        self._waterfall = new_waterfall.clip(0.0)
        self.profile_charge = self.calc_profilecharge(
                                self._waterfall[self.machine.beam_ref_frame])
        self._waterfall = (self._waterfall
                            / np.vstack(np.sum(self._waterfall, axis=1)))
        self.machine.nbins = self._waterfall.shape[1]
        log.info(f'Waterfall loaded. (profile charge: '
                 f'{self.profile_charge:.3E}, nbins: {self.machine.nbins})')

    def calc_self_fields(self, filtered_profiles=None):
        if filtered_profiles is None:
            self.dsprofiles = signal.savgol_filter(
                                x=self._waterfall, window_length=7,
                                polyorder=4, deriv=1)
        else:
            # Insert assertions.
            # Create spescialized exceptions.
            self.dsprofiles = filtered_profiles

        self.vself = self._calculate_self()

    
    # ============== END NEW ===============

    # Main function for the time space class
    # @profile
    def create(self, raw_data):
        
        # Converting from raw data to profiles.
        # Result is saved in self.profiles

        if self.machine.xat0 < 0:
            (self.fitted_xat0,
             self.tangentfoot_low,
             self.tangentfoot_up) = self.fit_xat0()
            self.machine.xat0 = self.fitted_xat0

        (self.phiwrap,
         self.wrap_length) = self.find_wrap_length()

        log.info(f'x at zero: {self.machine.xat0}')
        log.info(f'y at zero: {self.machine.yat0}')


    # Contains functions for calculations using the self-field voltage
    def _calc_using_self_field(self):
        log.info('Calculating self-fields')
        self.dsprofiles = signal.savgol_filter(
                            x=self.profiles, window_length=7,
                            polyorder=4, deriv=1)
        self.vself = self._calculate_self()

    # Perform at fit for finding x at 0
    def fit_xat0(self):
        ref_idx = self.machine.beam_ref_frame
        ref_prof = self.profiles[ref_idx]
        ref_turn = ref_idx * self.machine.dturns

        log.info(f'Performing fit for xat0 '
                     f'using reference profile: {ref_idx}')

        (tfoot_up,
         tfoot_low) = self.calc_tangentfeet(ref_prof)
        bunch_duration = (tfoot_up - tfoot_low) * self.machine.dtbin

        # bunch_phase_length is needed in phase_low function
        bunch_phaselength = (self.machine.h_num * bunch_duration
                             * self.machine.omega_rev0[ref_turn])
        log.info(f'Calculated bunch phase length: {bunch_phaselength}')

        # Find roots of phaselow function
        x0 = self.machine.phi0[ref_turn] - bunch_phaselength / 2.0
        phil = optimize.newton(func=phase_low,
                               x0=x0,
                               fprime=dphase_low,
                               tol=0.0001,
                               maxiter=100,
                               args=(self.machine, bunch_phaselength,
                                     ref_turn))
        fitted_xat0 = (tfoot_low + (self.machine.phi0[ref_turn] - phil)
                    / (self.machine.h_num
                       * self.machine.omega_rev0[ref_turn]
                       * self.machine.dtbin))
        log.info(f'Fitted x at zero: {fitted_xat0}')

        return fitted_xat0, tfoot_low, tfoot_up

    # Calculate self-field voltage (if self_field_flag is True)
    def _calculate_self(self):
        vself = np.zeros((self.machine.nprofiles - 1,
                          self.wrap_length + 1),
                         dtype=float)
        for i in range(self.machine.nprofiles - 1):
            vself[
                i, :self.machine.nbins] = (0.5 * self.profile_charge
                                       * (self.machine.sfc[i]
                                          * self.dsprofiles[
                                                i,:self.machine.nbins]
                                          + self.machine.sfc[i + 1]
                                          * self.dsprofiles[
                                                i + 1, :self.machine.nbins]))
        return vself

    # Calculate the total charge of profile
    def calc_profilecharge(self, ref_prof):
        return (np.sum(ref_prof) * self.machine.dtbin
                / (self.machine.rebin * e_UNIT
                   * self.machine.pickup_sensitivity))

    # return index of last bins to the left and right of max valued bin,
    # with value over the threshold.
    def _calc_tangentbins(self, ref_profile, threshold_coeff=0.15):
        threshold = threshold_coeff * np.max(ref_profile)

        maxbin = np.argmax(ref_profile)
        for ibin in range(maxbin, 0, -1):
            if ref_profile[ibin] < threshold:
                tangent_bin_low = ibin + 1
                break
        for ibin in range(maxbin, self.machine.nbins):
            if ref_profile[ibin] < threshold:
                tangent_bin_up = ibin - 1
                break

        return tangent_bin_up, tangent_bin_low

    # Find foot tangents of profile. Needed to estimate bunch duration
    # when performing a fit to find xat0
    def calc_tangentfeet(self, ref_prof):       
        index_array = np.arange(self.machine.nbins) + 0.5

        (tanbin_up,
         tanbin_low) = self._calc_tangentbins(ref_prof)

        [bl, al] = np.polyfit(index_array[tanbin_low - 2:
                                          tanbin_low + 2],
                              ref_prof[tanbin_low - 2:
                                       tanbin_low + 2],
                              deg=1)

        [bu, au] = np.polyfit(index_array[tanbin_up - 1:
                                          tanbin_up + 3],
                              ref_prof[tanbin_up - 1:
                                       tanbin_up + 3],
                              deg=1)

        tanfoot_low = -1 * al / bl
        tanfoot_up = -1 * au / bu

        log.info(f'tangent_foot_low: {tanfoot_low:0.7f}, '
                     f'tangent_foot_up: {tanfoot_up:0.7f}')
        return tanfoot_up, tanfoot_low

    # Calculate the number of bins in the first
    # integer number of rf periods, larger than the image width.
    # -f: fortran compensation
    def find_wrap_length(self):
        if self.machine.bdot > 0.0:
            last_turn_index = ((self.machine.nprofiles - 1)
                               * self.machine.dturns - 1)
            drad_bin = (self.machine.h_num
                        * self.machine.omega_rev0[last_turn_index]
                        * self.machine.dtbin)
        else:
            drad_bin = (self.machine.h_num
                        * self.machine.omega_rev0[0]
                        * self.machine.dtbin)

        phiwrap = np.ceil(self.machine.nbins * drad_bin
                          / (2 * np.pi)) * 2 * np.pi

        wrap_length = int(np.ceil(phiwrap / drad_bin))

        log.info(f'phi wrap =  {str(phiwrap)}, '
                      f'wrap length =  {str(wrap_length)}')
        return phiwrap, wrap_length
