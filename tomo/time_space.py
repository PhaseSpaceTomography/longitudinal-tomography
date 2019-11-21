import logging
from scipy import signal
from scipy import optimize
import numpy as np
from machine import Machine
from physics import phase_low, dphase_low, e_UNIT
from utils.assertions import assert_equal, assert_inrange, assert_greater
from utils.exceptions import (RawDataImportError, InputError,
                               RebinningError)
# ================
# About TimeSpace:
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


class TimeSpace:

    def __init__(self, machine):
        self.machine = machine
        self.profiles = np.array([])
        self.profile_charge = None   # Total charge in profile

        # Self field variables:
        self.vself = None            # Self-field voltage
        self.dsprofiles = None       # Smoothed derivative of profiles

        self.tangentfoot_low = 0.0
        self.tangentfoot_up = 0.0
        self.phiwrap = 0.0
        self.wrap_length = 0
        self.fitted_xat0 = 0.0

    # Main function for the time space class
    # @profile
    def create(self, raw_data):
        
        # Converting from raw data to profiles.
        # Result is saved in self.profiles
        (self.profiles,
         self.profile_charge) = self.create_profiles(raw_data)

        if self.machine.xat0 < 0:
            (self.fitted_xat0,
             self.tangentfoot_low,
             self.tangentfoot_up) = self.fit_xat0()
            self.machine.xat0 = self.fitted_xat0

        (self.phiwrap,
         self.wrap_length) = self.find_wrap_length()

        logging.info(f'x at zero: {self.machine.xat0}')
        logging.info(f'y at zero: {self.machine.yat0}')

        if self.machine.self_field_flag:
            self._calc_using_self_field()

    # Creating profiles, updates profile length and calculates profile charge.
    def create_profiles(self, raw_data):

        # Asserting that the correct amount of measurement-data is provided.
        assert_equal(len(raw_data), 'length of raw_data',
                     self.machine.all_data, RawDataImportError)

        # Subtracting baseline from raw data
        raw_data = self.subtract_baseline(raw_data)

        # Splitting up raw data to profiles
        profiles = self.rawdata_to_profiles(raw_data)

        # Rebinning
        if self.machine.rebin > 1:
            profiles, self.machine.nbins = self.rebin(profiles)

        # Setting negative numbers to zero.
        profiles = profiles.clip(0.0)

        ref_prof = self.machine.beam_ref_frame - 1
        profile_charge = self.calc_profilecharge(profiles[ref_prof])

        profiles = self.normalize_profiles(profiles)

        logging.info('Profiles created successfully.')

        return profiles, profile_charge


    # Contains functions for calculations using the self-field voltage
    def _calc_using_self_field(self):
        logging.info('Calculating self-fields')
        self.dsprofiles = signal.savgol_filter(
                            x=self.profiles, window_length=7,
                            polyorder=4, deriv=1)

        self.vself = self._calculate_self()

    # Perform at fit for finding x at 0
    def fit_xat0(self):
        ref_idx = self.machine.beam_ref_frame - 1
        ref_prof = self.profiles[ref_idx]
        ref_turn = ref_idx * self.machine.dturns

        logging.info(f'Performing fit for xat0 '
                     f'using reference profile: {ref_idx}')

        (tfoot_up,
         tfoot_low) = self.calc_tangentfeet(ref_prof)
        bunch_duration = (tfoot_up - tfoot_low) * self.machine.dtbin

        # bunch_phase_length is needed in phase_low function
        bunch_phaselength = (self.machine.h_num * bunch_duration
                             * self.machine.omega_rev0[ref_turn])
        logging.info(f'Calculated bunch phase length: {bunch_phaselength}')

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
        logging.info(f'Fitted x at zero: {fitted_xat0}')

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

    # Original function for subtracting baseline of raw data input profiles.
    # Finds the baseline from the first 5% (by default)
    #  of the beam reference profile.
    def subtract_baseline(self, raw_data, percentage=0.05):
        istart = int((self.machine.frame_skipcount
                      + self.machine.beam_ref_frame - 1)
                     * self.machine.framelength + self.machine.preskip_length)

        assert_inrange(percentage, 'percentage',
                       0.0, 1.0, InputError,
                       'The chosen percentage of raw_data '
                       'to create baseline from is not valid')

        iend = int(np.floor(percentage * self.machine.nbins
                            + istart + 1))

        baseline = (np.sum(raw_data[istart:iend])
                    / np.real(np.floor(percentage
                                       * self.machine.nbins + 1)))
        
        logging.info(f'A baseline was found with the value: {str(baseline)}')
        
        return raw_data - baseline

    # Turns list of raw data into (profile count, profile length) shaped array.
    def rawdata_to_profiles(self, raw_data):
        profiles = raw_data.reshape((self.machine.nprofiles,
                                     self.machine.framelength))
        
        if self.machine.postskip_length > 0:
            profiles = profiles[:, self.machine.preskip_length:
                                   -self.machine.postskip_length]
        else:
            profiles = profiles[:, self.machine.preskip_length:]
        
        assert_equal(profiles.shape[1], 'profile length',
                     self.machine.nbins, InputError,
                     'raw data was reshaped to profiles with '
                     'a wrong shape.')

        logging.debug(f'{self.machine.nprofiles} profiles '
                      f'with length {self.machine.nbins} '
                      f'created from raw data')

        return profiles


    # Re-binning of profiles from original number of bins to smaller number of
    # bins specified in input parameters
    def rebin(self, profiles):
        # Find new profile length
        if self.machine.nbins % self.machine.rebin == 0:
            new_prof_len = int(self.machine.nbins / self.machine.rebin)
        else:
            new_prof_len = int(self.machine.nbins / self.machine.rebin) + 1

        assert_greater(new_prof_len,
                       'rebinned profile length', 1,
                       RebinningError,
                       f'The length of the profiles after re-binning'
                       f'is not valid...\nMake sure that the re-binning '
                       f'factor ({self.machine.rebin}) is not larger than'
                       f'the original profile length '
                       f'({self.machine.nbins})')

        # Re-binning profiles until second last bin
        new_profilelist = np.zeros((self.machine.nprofiles, new_prof_len))
        for p in range(self.machine.nprofiles):
            for i in range(new_prof_len - 1):
                binvalue = 0.0
                for bincounter in range(self.machine.rebin):
                    binvalue += profiles[p, i * self.machine.rebin
                                            + bincounter]
                new_profilelist[p, i] = binvalue

        # Re-binning last profile bins
        for p in range(self.machine.nprofiles):
            binvalue = 0.0
            for i in range((new_prof_len - 1) * self.machine.rebin,
                           self.machine.nbins):
                binvalue += profiles[p, i]
            binvalue *= (float(self.machine.rebin)
                         / float(self.machine.nbins
                                 - (new_prof_len - 1)
                                 * self.machine.rebin))
            new_profilelist[p, -1] = binvalue

        logging.info('Profile rebinned with a rebin factor of '
                     + str(self.machine.rebin))

        return new_profilelist, new_prof_len

    def normalize_profiles(self, profiles):
        return profiles / np.vstack(np.sum(profiles, axis=1))

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

        logging.info(f'tangent_foot_low: {tanfoot_low:0.7f}, '
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

        logging.info(f'phi wrap =  {str(phiwrap)}, '
                      f'wrap length =  {str(wrap_length)}')
        return phiwrap, wrap_length
