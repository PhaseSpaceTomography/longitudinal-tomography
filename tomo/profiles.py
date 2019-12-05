import logging as log
import numpy as np
import scipy.signal as sig
from scipy import constants

from . import physics
from .utils import exceptions as expt

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

    def __init__(self, machine, sampling_time, waterfall,
                 profile_charge=None):
        self.machine = machine
        self.sampling_time = sampling_time
        
        self.waterfall = waterfall

        self.profile_charge = profile_charge


    @property
    def waterfall(self):
        return self._waterfall

    @waterfall.setter
    def waterfall(self, new_waterfall):
        log.info('Loading waterfall...')
        
        if not hasattr(new_waterfall, '__iter__'):
            raise expt.WaterfallError('waterfall should be an iterable.')

        if not new_waterfall.shape[0] == self.machine.nprofiles:
            err_msg = f'Waterfall does not correspond to machine object.\n'\
                      f'Expected nr of profiles: {self.machine.nprofiles}, '\
                      f'nr of profiles in waterfall: {new_waterfall.shape[0]}'
            raise expt.WaterfallError(err_msg)

        new_waterfall = np.array(new_waterfall)
        self._waterfall = new_waterfall.clip(0.0)
        if np.sum(np.abs(self.waterfall)) == 0.0:
            raise expt.WaterfallReducedToZero()
        
        self.machine.nbins = self._waterfall.shape[1]
        log.info(f'Waterfall loaded with shape: {self.waterfall.shape})')

    # Calculate the total charge of profile.
    # Uses the beam reference profile for the calculation
    def calc_profilecharge(self):
        ref_prof = self.waterfall[self.machine.beam_ref_frame]
        self.profile_charge = (np.sum(ref_prof) * self.sampling_time
                               / (constants.e
                                 * self.machine.pickup_sensitivity))

    # Calculate self-fields based on filtered profiles.
    # If filtered profiles are not provided by the user,
    # standard filter (savitzky-golay smoothing filter) is used.
    def calc_self_fields(self, filtered_profiles=None):
        if self.profile_charge is None:
            err_msg = 'Profile charge must be calculated before '\
                      'calculating the self-fields'
            raise expt.ProfileChargeNotCalculated(err_msg)

        if filtered_profiles is None:
            self.dsprofiles = sig.savgol_filter(
                                x=self._waterfall, window_length=7,
                                polyorder=4, deriv=1)
        else:
            self.dsprofiles = self._check_manual_filtered_profs(
                                                    filtered_profiles)

        (self.phiwrap,
         self.wrap_length) = self._find_wrap_length()

        self.vself = self._calculate_self()
        log.info('Self fields were calculated.')

    def _check_manual_filtered_profs(self, fprofs):
        if not hasattr(fprofs, '__iter__'):
            err_msg = 'Filtered profiles should be iterable.'
            raise expt.FilteredProfilesError(err_msg)
        fprofs = np.array(fprofs)
        if fprofs.shape == self.waterfall.shape:
            return fprofs
        else:
            err_msg = f'Filtered profiles should have the same shape'\
                      f'as the waterfall.\n'\
                      f'Shape profiles: {fprofs.shape}\n'\
                      f'Shape waterfall: {self.waterfall.shape}\n'
            raise expt.FilteredProfilesError(err_msg)

    # Calculate the number of bins in the first
    # integer number of rf periods, larger than the image width.
    def _find_wrap_length(self):
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

    # Calculate self-field voltages
    def _calculate_self(self):
        sfc = physics.calc_self_field_coeffs(self.machine)
        vself = np.zeros((self.machine.nprofiles - 1,
                          self.wrap_length + 1),
                         dtype=float)
        for i in range(self.machine.nprofiles - 1):
            vself[i, :self.machine.nbins] = (0.5 * self.profile_charge
                                             * (sfc[i]
                                             * self.dsprofiles[
                                                i,:self.machine.nbins]
                                             + sfc[i + 1]
                                             * self.dsprofiles[
                                                i + 1, :self.machine.nbins]))
        return vself
