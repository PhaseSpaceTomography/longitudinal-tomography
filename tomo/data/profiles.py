"""Module containing the Profiles class for storing measurements

:Author(s): **Christoffer Hjertø Grindheim**
"""

import logging as log
import numpy as np
import scipy.signal as sig
from scipy import constants

from ..utils import physics
from ..utils import exceptions as expt

class Profiles(object):
    '''Class holding measured data.

    This class holds the measured data (waterfall) and their properties.

    The waterfall is a nprofiles x nbins shaped array, waterfall[0] is the
    first measurement. Each bin is `machine.dtbin` long,
    and holds the intensity of the beam at this time of the measurement.

    The profile charge can be provided, or calculated using
    the `calc_profilecharge` function which is based
    on the machines pickup sensitivity.

    The class can also be used to calculate the self-fields of the bunch.

    Parameters
    ----------
    machine: Machine
        Used to import different quantities like turns between measurements.
    sampling_time: float
        Original measurement sampling time.
    waterfall: ndarray, float
        2D-array containing all profile measurements.
    profile_charge: float
        Total charge of a reference profile.

    Attributes
    ----------
    machine: Machine
        The machine and its settings when measurements was taken.
    sampling_time: float
        Original sampling time of measurements.
        Needed for calculation of profile charge.
    waterfall: ndarray, float
        2D-array containing all profile measurements.
    profile_charge: float
        Total charge of a reference profile.
    vself: ndarray, float
        2D array of self-fields at each bin of each profile.
    dsprofiles: ndarray, float
        Filtered profiles.
    phiwrap: float
        Phase covering an integer of rf periods.
    wrap_length: float
        Maximum number of bins to cover an integer number of rf periods.

    '''
    def __init__(self, machine, sampling_time, waterfall,
                 profile_charge=None):
        self.machine = machine
        self.sampling_time = sampling_time
        
        self.waterfall = waterfall

        self.profile_charge = profile_charge

    @property
    def waterfall(self):
        '''Waterfall defined as @property.

        * Asserts that input waterfall is iterable and has the correct\
        amount of profiles, as stated by the Machine object.
        * Removes negative values from waterfall (set to zero).
        * Updates machine.nbins.


        Parameters
        ----------
        waterfall: ndarray, float
            Measured profiles as a 2D array.
        
        Returns
        -------
        waterfall: ndarray, float
            Measured profiles as a 2D array.

        Raises
        ------
        WaterfallError: Exception
            If not iterable or wrong amount of profiles.
        WaterfallReducedToZero: Exception
            If all of profile is redduced to zero after removing
            negaitve values.
        '''
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

    def calc_profilecharge(self):
        '''Calculate the total charge of profile.
        Uses the beam reference profile for the calculation.
        Sets the `profile_charge` field.
        '''
        ref_prof = self.waterfall[self.machine.beam_ref_frame]
        self.profile_charge = (np.sum(ref_prof) * self.sampling_time
                               / (constants.e
                                 * self.machine.pickup_sensitivity))

    def calc_self_fields(self, filtered_profiles=None, in_filter=None):
        '''Calculate self-fields based on filtered profiles.
        If filtered profiles are not provided by the user,
        standard filter (savitzky-golay smoothing filter) is used.

        The function sets the following fields:
        
        * vself - 2D array of self-fields at each bin of each profile.
        * dsprofiles - Filtered profiles.
        * phiwrap - Phase covering an integer of rf periods.
        * wrap_length - Maximum number of bins to cover\
                        an integer number of rf periods.

        A description of these attributes can be found in the documentation\
        for the `Profiles` class.

        Parameters
        ----------
        filtered_profiles: ndarray, float
            If filtered profiles are provided, they will be used in the\
            calculation of the self fields. If not, the original profiles\
            will be filtered using a standard or user spescified filter.
        in_filter: function
            The measured profiles will be filtered using the provided filter\
            in stead of the savitzky-golay smoothing filter.

        Raises
        ------
        ProfileChargeNotCalculated: Exception
            No profile charge has been provided or calculated.
        FilteredProfilesError: Exception
            Filtered profiles has the wrong shape or are not iterable.
        '''
        if self.profile_charge is None:
            err_msg = 'Profile charge must be calculated before '\
                      'calculating the self-fields'
            raise expt.ProfileChargeNotCalculated(err_msg)

        if filtered_profiles is not None:
            self.dsprofiles = self._check_manual_filtered_profs(
                                                    filtered_profiles)
        elif in_filter is not None:
            self.dsprofiles = np.copy(self.waterfall)
            self.dsprofiles = in_filter(self.dsprofiles)
            self.dsprofiles = self._check_manual_filtered_profs(
                                                    self.dsprofiles)
        else:
            self.dsprofiles = np.copy(self.waterfall)
            # Makes normalized version of waterfall
            self.dsprofiles /= np.sum(self.dsprofiles, axis=1)[:, None]
            self.dsprofiles = sig.savgol_filter(
                                x=self.dsprofiles, window_length=7,
                                polyorder=4, deriv=1)


        (self.phiwrap,
         self.wrap_length) = self._find_wrap_length()

        self.vself = self._calculate_self()
        log.info('Self fields were calculated.')

    # Checks the filtered profiles
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
