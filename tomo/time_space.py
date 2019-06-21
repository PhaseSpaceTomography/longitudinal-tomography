import logging
import scipy.signal._savitzky_golay as savgol
import numpy as np
from parameters import Parameters
import physics
from numeric import newton
from utils.assertions import TomoAssertions as ta
from utils.assertions import (RawDataImportError,
                              InputError,
                              RebinningError)


# The TimeSpace class handles import and processing of data in time domain.
#  - importing raw data, and converting it to profiles, specified by input parameters.
#  - calculation of total charge in profile
#  - if xat0, from input parameters, is less than 0 wil a fit be performed to find xat0
#  - calculations of yat0, phi wrap, xorigin
#  - if self field flag is true will self fields be included in the reconstruction.
#       in the time space class will a savitzky-golay smoothing filter be applied to the profiles
#       and the self field voltage will be calculated.

# Variables used in this class are retrieved from a parameter object,
#   which stores the input parameters for the reconstruction.
#   A description of all the input and time space variables can be found in the parameters module.

# TimeSpace variables:
# ===================
# par               parameter object (for more info, see parameters module)
# profiles          a profile-count * profile-length sized matrix containing profile data, ready for reconstruction.
# profile_charge    Total charge in reference profile
#
# Self field variables:
# ---------------------
# vself             Self-field voltage
# dsprofiles        Smoothed derivative of profiles
#
# saved in parameters:
# - - - - - - - - - - -
# bunch_phaselength Bunch phase length in beam reference profile
# tangentfoot_low   Used for estimation of bunch duration
# tangentfoot_up
# phiwrap           Phase covering the an integer of rf periods
# wrap_length       Maximum number of bins to cover an integer number of rf periods
# fit_xat0          Value of (if) fitted xat0
# x_origin = 0.0    absolute difference in bins between phase=0
#                       and origin of the reconstructed phase-space coordinate system.
#


class TimeSpace:

    def __init__(self, parameter_file_path):
        self.par = Parameters()
        self.profiles = np.array([])
        self.profile_charge = None	 # Total charge in profile

        # Self field variables:
        self.vself = None            # Self-field voltage
        self.dsprofiles = None       # Smoothed derivative of profiles

        self.run_time_space(parameter_file_path)

    # Main function for the time space class
    def run_time_space(self, parameter_file_path):
        self.par.get_parameters_txt(parameter_file_path)
        self.get_profiles(parameter_file_path)
        self.adjust_profiles()

        self.profile_charge = self.total_profilecharge(
                                self.profiles[self.par.beam_ref_frame - 1],
                                self.par.dtbin, self.par.rebin,
                                self.par.pickup_sensitivity)

        self.profiles = self.normalize_profiles(self.profiles)

        if self.par.xat0 < 0:
            idx = self.par.beam_ref_frame - 1
            (self.par.fit_xat0,
             self.par.tangentfoot_low,
             self.par.tangentfoot_up,
             self.par.bunch_phaselength) = self._fit_xat0(
                                                self.profiles[idx, :], idx)
            self.par.xat0 = self.par.fit_xat0

        self.par.x_origin = self.calc_xorigin(
                                self.par.beam_ref_frame - 1, self.par.dturns,
                                self.par.phi0, self.par.h_num,
                                self.par.omega_rev0, self.par.dtbin,
                                self.par.xat0)

        (self.par.phiwrap,
         self.par.wrap_length) = self._find_wrap_length(
                                    self.par.profile_count, self.par.dturns,
                                    self.par.dtbin, self.par.h_num,
                                    self.par.omega_rev0,
                                    self.par.profile_length, self.par.bdot)

        self.par.yat0 = self._find_yat0(self.par.profile_length)

        if self.par.self_field_flag:
            self._calc_using_self_field()

    # Subroutine for reading raw data (from text file), subtracting baseline
    # and splitting the raw data to profiles.
    def get_profiles(self, parameter_file_path):

        # collecting profile raw data
        data = self.get_indata_txt(self.par.rawdata_file,
                                   parameter_file_path)

        # Subtracting baseline from raw data
        data = self.subtract_baseline(
                        data, self.par.frame_skipcount,
                        self.par.beam_ref_frame, self.par.framelength,
                        self.par.preskip_length, self.par.profile_length)

        # Splitting up raw data to profiles
        self.profiles = self.rawdata_to_profiles(
                            data, self.par.profile_count,
                            self.par.profile_length, self.par.frame_skipcount,
                            self.par.framelength, self.par.preskip_length,
                            self.par.postskip_length)

    # Subroutine for re-binning profiles and converting negative values to zeros.
    def adjust_profiles(self):
        if self.par.rebin > 1:
            (self.profiles,
             self.par.profile_length) = self.rebin(self.profiles,
                                                   self.par.rebin,
                                                   self.par.profile_length,
                                                   self.par.profile_count)

        self.profiles = self.negative_profile_data_zero(self.profiles)

    # Contains functions for calculations using the self-field voltage
    def _calc_using_self_field(self):
        logging.info("Calculating self-fields")
        self.dsprofiles = self._filter()
        self.vself = self._calculate_self()

    # Perform at fit for finding x at 0
    # Set bunch_phaselength variable, needed for phaselow function
    def _fit_xat0(self, profile, refprofile_index, threshold_value=0.15):
        logging.info("Performing fit for xat0")

        tangentfoot_up, tangentfoot_low = self._calc_tangentfeet(
                                                profile, refprofile_index,
                                                self.par.profile_length,
                                                threshold_value)

        thisturn = refprofile_index * self.par.dturns

        bunch_duration = (tangentfoot_up - tangentfoot_low) * self.par.dtbin

        # bunch_phase_length is needed in phaselow
        self.par.bunch_phaselength = (self.par.h_num * bunch_duration
                                      * self.par.omega_rev0[thisturn])

        # Find roots of phaselow function
        xstart_newt = self.par.phi0[thisturn] - self.par.bunch_phaselength / 2.0

        # phaselow and dPhaseLow are functions from Physics module
        phil = newton(physics.phase_low, physics.dphase_low, xstart_newt,
                      self.par, thisturn, 0.0001)

        fit_xat0 = (tangentfoot_low + (self.par.phi0[thisturn] - phil)
                    / (self.par.h_num
                       * self.par.omega_rev0[thisturn]
                       * self.par.dtbin))

        return fit_xat0, tangentfoot_low, tangentfoot_up, self.par.bunch_phaselength

    # Savitzky-Golay smoothing filter (if self_field_flag is True)
    def _filter(self):
        filtered_profiles = savgol.savgol_filter(x=self.profiles,
                                                 window_length=7,
                                                 polyorder=4,
                                                 deriv=1)
        return filtered_profiles

    # Calculate self-field voltage (if self_field_flag is True)
    def _calculate_self(self):
        vself = np.zeros((self.par.profile_count - 1,
                          self.par.wrap_length),
                         dtype=float)
        for i in range(self.par.profile_count - 1):
            vself[i, 0:self.par.profile_length]\
                = (0.5
                   * self.profile_charge
                   * (self.par.sfc[i]
                      * self.dsprofiles[i, :self.par.profile_length]
                      + self.par.sfc[i + 1]
                      * self.dsprofiles[i + 1, :self.par.profile_length]))
        return vself

    # Write files to text file.
    def save_profiles_text(self, profiles, output_directory, filename):
        np.savetxt(output_directory + filename, profiles.flatten().T)
        logging.info("Saved profiles to: " + output_directory + filename)

    # Read data from separate text file or as extension of parameter file
    @staticmethod
    def get_indata_txt(filename, par_file_path, skiplines=98):
        if filename != "pipe":
            inn_data = np.genfromtxt(filename, dtype=float)
        else:
            inn_data = np.genfromtxt(par_file_path,
                                     skip_header=skiplines, dtype=float)
        ta.assert_greater(len(inn_data),
                          'number of importedraw data elements',
                          0, RawDataImportError,
                          f'No raw data was found in file: {filename}')
        return inn_data

    # Original function for subtracting baseline of raw data input profiles.
    @staticmethod
    def subtract_baseline(data, frame_skipcount, beam_ref_frame, frame_length,
                          preskip_length, profile_length, percentage=0.05):
        # Find the baseline from the first 5% (by default) of
        # beam reference profile.
        i0 = int((frame_skipcount + beam_ref_frame - 1) * frame_length
                 + preskip_length)

        ta.assert_inrange(percentage, 'percentage', 0.0, 1.0, InputError,
                          'The chosen percentage of data to create baseline'
                          'from is not valid')

        i_five_percent = int(np.floor(i0 + percentage * profile_length + 1))
        baseline = (np.sum(data[i0:i_five_percent])
                    / np.real(np.floor(percentage * profile_length + 1)))

        logging.debug(f"A baseline was found with the value: {str(baseline)}")
        return data - baseline

    # Turns list of raw data into (profile count, profile length) shaped array.
    @staticmethod
    def rawdata_to_profiles(data, profile_count, profile_length,
                            frame_skipcount, framelength, preskip_length,
                            postskip_length):
        profiles = np.zeros((profile_count, profile_length))
        for i in range(profile_count):
            profile_start = ((frame_skipcount + i) * framelength
                             + preskip_length)
            profile_end = ((frame_skipcount + i + 1) * framelength
                           - postskip_length)
            profiles[i, :] = data[profile_start:profile_end]
        logging.info(f'{str(profile_count)} profiles '
                     f'with length {profile_length} '
                     f'created from raw data')
        return profiles

    # Re-binning of profiles from original number of bins to smaller number of
    # bins specified in input parameters
    @staticmethod
    def rebin(profiles, rebin_factor, profile_length, profile_count):
        # Find new profile length
        if profile_length % rebin_factor == 0:
            new_profile_length = int(profile_length / rebin_factor)
        else:
            new_profile_length = int(profile_length / rebin_factor) + 1

        ta.assert_greater(new_profile_length,
                          'rebinned profile length', 1,
                          RebinningError,
                          f'The length of the profiles after re-binning'
                          f'is not valid...\nMake sure that the re-binning '
                          f'factor ({rebin_factor}) is not larger than'
                          f'the original profile length ({profile_length})')

        # Re-binning profiles until second last bin
        new_profilelist = np.zeros((profile_count, new_profile_length))
        for p in range(profile_count):
            for i in range(new_profile_length - 1):
                binvalue = 0.0
                for bincounter in range(rebin_factor):
                    binvalue += profiles[p, i * rebin_factor + bincounter]
                new_profilelist[p, i] = binvalue

        # Re-binning last profile bins
        for p in range(profile_count):
            binvalue = 0.0
            for i in range((new_profile_length - 1) * rebin_factor, profile_length):
                binvalue += profiles[p, i]
            binvalue *= (float(rebin_factor)
                         / float(profile_length
                                 - (new_profile_length - 1)
                                 * rebin_factor))
            new_profilelist[p, -1] = binvalue

        logging.info("Profile rebinned with a rebin factor of "
                     + str(rebin_factor))

        return new_profilelist, new_profile_length

    # Setting all negative profile data to zero
    @staticmethod
    def negative_profile_data_zero(profiles):
        profiles = np.where(profiles < 0, 0, profiles)
        ta.assert_array_not_equal(profiles,
                                  'profile without negative numbers', 0,
                                  'The whole profile was reduced to zeroes '
                                  'when changing negative numbers to zeroes.')
        return profiles

    # Normalize profile data to number between 0 and 1
    @staticmethod
    def normalize_profiles(profiles):
        return profiles / np.vstack(np.sum(profiles, axis=1))

    # Calculate the total charge of profile
    @staticmethod
    def total_profilecharge(ref_prof, dtbin, rebin, pickup_sens):
        return np.sum(ref_prof) * dtbin / (rebin * physics.e_UNIT * pickup_sens)

    # Calculate the absolute difference (in bins) between phase=0 and
    # origin of the reconstructed phase space coordinate system.
    @staticmethod
    def calc_xorigin(beam_ref_frame, dturns, phi0,
                     h_num, omega_rev0, dtbin, xat0):
        reference_turn = beam_ref_frame * dturns
        x_origin = (phi0[reference_turn]
                    / (h_num
                       * omega_rev0[reference_turn]
                       * dtbin)
                    - xat0)

        logging.debug("xat0: " + str(xat0) + ", x_origin: " + str(x_origin))

        return x_origin

    # return index of last bins to the left and right of max valued bin,
    # with value over the threshold.
    @staticmethod
    def _calc_tangentbins(profile, profile_length, threshold):
        maxbin = np.argmax(profile)
        for ibin in range(maxbin, 0, -1):
            if profile[ibin] < threshold:
                tangent_bin_low = ibin + 1
                break
        for ibin in range(maxbin, profile_length):
            if profile[ibin] < threshold:
                tangent_bin_up = ibin - 1
                break

        return tangent_bin_up, tangent_bin_low

    # Find foot tangents of profile. Needed to estimate bunch duration
    # when performing a fit to find xat0
    @staticmethod
    def _calc_tangentfeet(profile, refprofile_index,
                          profile_length, threshold_value):
        index_array = np.arange(profile_length, dtype=float) + 0.5

        threshold = threshold_value * np.max(profile.data)

        (tangent_bin_up,
         tangent_bin_low) = TimeSpace._calc_tangentbins(
                                    profile, profile_length, threshold)

        logging.info("findxat0: beam.ref.indx.: " + str(refprofile_index)
                     + ", threshold: " + str(threshold))

        [bl, al] = np.polyfit(index_array[tangent_bin_low - 2:
                                          tangent_bin_low + 2],
                              profile[tangent_bin_low - 2:
                                      tangent_bin_low + 2],
                              deg=1)

        [bu, au] = np.polyfit(index_array[tangent_bin_up - 1:
                                          tangent_bin_up + 3],
                              profile[tangent_bin_up - 1:
                                      tangent_bin_up + 3],
                              deg=1)

        tangentfoot_low = -1 * al / bl
        tangentfoot_up = -1 * au / bu

        logging.debug("tangent_foot_low = " + str(tangentfoot_low)
                      + ", tangent_foot_up = " + str(tangentfoot_up))

        return tangentfoot_up, tangentfoot_low

    # Calculate the number of bins in the first
    # integer number of rf periods, larger than the image width.
    @staticmethod
    def _find_wrap_length(profile_count, dturns, dtbin, h_num, omega_rev0,
                          profile_length, bdot):
        if bdot > 0.0:
            last_turn_index = ((profile_count - 1) * dturns - 1)
            drad_bin = h_num * omega_rev0[last_turn_index] * dtbin
        else:
            drad_bin = h_num * omega_rev0[0] * dtbin

        phiwrap = (np.ceil(profile_length * drad_bin / (2*np.pi)) * 2 * np.pi)
        wrap_length = int(np.ceil(phiwrap / drad_bin))

        logging.debug(f"findwrap_length: phiwrap =  {str(phiwrap)},"
                      f" wrap_length =  {str(wrap_length)}")
        return phiwrap, wrap_length

    @staticmethod
    def _find_yat0(profile_length):
        yat0 = profile_length / 2.0
        logging.debug("yat0 = " + str(yat0))
        return yat0
