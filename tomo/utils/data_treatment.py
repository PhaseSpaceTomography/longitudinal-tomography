import numpy as np
from scipy import optimize

from . import assertions as asrt
from . import exceptions as expt
from .. import physics

# Original function for subtracting baseline of raw data input profiles.
# Finds the baseline from the first 5% (by default)
#  of the beam reference profile.
def calc_baseline_ftn(waterfall, ref_prof, percent=0.05):
    asrt.assert_inrange(percent, 'percent', 0.0, 1.0, expt.InputError,
                        'The chosen percent of raw_data '
                        'to create baseline from is not valid')

    nbins = len(waterfall[ref_prof])
    iend = int(percent * nbins) 

    return np.sum(waterfall[ref_prof, :iend]) / np.floor(percent * nbins)


def rebin(waterfall, rbn, machine=None, dtbin=None):
    data = np.copy(waterfall)

    # Check that there is enough data to for the given rebin factor.
    if data.shape[1] % rbn == 0:
        rebinned = _rebin_dividable(data, rbn)
    else:
        rebinned = _rebin_individable(data, rbn)

    if machine is not None:
        machine.dtbin *= rbn
        machine.synch_part_x /= float(rbn)

    if dtbin is not None:
        return rebinned, dtbin * rbn
    else:
        return rebinned


# Rebins an 2d array given a rebin factor (rbn).
# The given array MUST have a length equal to an even number.
def _rebin_dividable(data, rbn):
    if data.shape[1] % rbn != 0:
        raise AssertionError('Input array must be '
                             'dividable on the rebin factor.')
    ans = np.copy(data)
    
    nprofs = data.shape[0]
    nbins = data.shape[1]

    new_nbins = int(nbins / rbn)
    all_bins = new_nbins * nprofs
    
    ans = ans.reshape((all_bins, rbn))
    ans = np.sum(ans, axis=1)
    ans = ans.reshape((nprofs, new_nbins))

    return ans


# Rebins an 2d array given a rebin factor (rbn).
# The given array MUST have vector length equal to an odd number.
def _rebin_individable(data, rbn):
    nprofs = data.shape[0]
    nbins = data.shape[1]

    ans = np.zeros((nprofs, int(nbins / rbn) + 1))

    last_data_idx = int(nbins / rbn) * rbn
    ans[:,:-1] = _rebin_dividable(data[:,:last_data_idx], rbn)
    ans[:,-1] = _rebin_last(data, rbn)[:, 0]
    return ans


# Rebins last indices of an 2d array given a rebin factor (rbn).
# Needed for the rebinning of odd arrays.
def _rebin_last(data, rbn):
    nprofs = data.shape[0]
    nbins = data.shape[1]

    i0 = (int(nbins / rbn) - 1) * rbn
    ans = np.copy(data[:,i0:])
    ans = np.sum(ans, axis=1)
    ans[:] *= rbn / (nbins - i0)
    ans = ans.reshape((nprofs, 1))
    return ans


# Original function for finding synch_part_x
# Finds synch_part_x based on a linear fit on a refence profile.  
def fit_synch_part_x(profiles):
    ref_idx = profiles.machine.beam_ref_frame
    ref_prof = profiles.waterfall[ref_idx] 
    ref_turn = ref_idx * profiles.machine.dturns

    tfoot_up, tfoot_low = _calc_tangentfeet(ref_prof)
    bunch_duration = (tfoot_up - tfoot_low) * profiles.machine.dtbin
    
    bunch_phaselength = (profiles.machine.h_num * bunch_duration
                         * profiles.machine.omega_rev0[ref_turn])

    x0 = profiles.machine.phi0[ref_turn] - bunch_phaselength / 2.0
    phil = optimize.newton(
            func=physics.phase_low, x0=x0,
            fprime=physics.dphase_low,
            tol=0.0001, maxiter=100,
            args=(profiles.machine, bunch_phaselength, ref_turn))
    
    fitted_synch_part_x = (tfoot_low + (profiles.machine.phi0[ref_turn] - phil)
                           / (profiles.machine.h_num
                           * profiles.machine.omega_rev0[ref_turn]
                           * profiles.machine.dtbin))

    return (fitted_synch_part_x, tfoot_low, tfoot_up)


# Find foot tangents of profile. Needed to estimate bunch duration
# when performing a fit to find synch_part_x
def _calc_tangentfeet(ref_prof):       
    nbins = len(ref_prof)
    index_array = np.arange(nbins) + 0.5

    tanbin_up, tanbin_low = _calc_tangentbins(ref_prof, nbins)

    [bl, al] = np.polyfit(index_array[tanbin_low - 2: tanbin_low + 2],
                          ref_prof[tanbin_low - 2: tanbin_low + 2], deg=1)

    [bu, au] = np.polyfit(index_array[tanbin_up - 1: tanbin_up + 3],
                          ref_prof[tanbin_up - 1: tanbin_up + 3], deg=1)

    tanfoot_low = -1 * al / bl
    tanfoot_up = -1 * au / bu

    return tanfoot_up, tanfoot_low


# return index of last bins to the left and right of max valued bin,
# with value over the threshold.
def _calc_tangentbins(ref_profile, nbins, threshold_coeff=0.15):
    threshold = threshold_coeff * np.max(ref_profile)
    maxbin = np.argmax(ref_profile)
    for ibin in range(maxbin, 0, -1):
        if ref_profile[ibin] < threshold:
            tangent_bin_low = ibin + 1
            break
    for ibin in range(maxbin, nbins):
        if ref_profile[ibin] < threshold:
            tangent_bin_up = ibin - 1
            break

    return tangent_bin_up, tangent_bin_low