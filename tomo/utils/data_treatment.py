'''Module containing functions for threatement of data.

:Author(s): **Christoffer Hjertø Grindheim**
'''

import numpy as np
from scipy import optimize

from . import assertions as asrt
from . import exceptions as expt
from . import physics


def calc_baseline_ftn(waterfall, ref_prof, percent=0.05):
    '''Function for finding baseline of raw data.
    
    The function is based on the original Fortran program,
    and uses a percentage of a reference profile in order to
    find the baseline of the measurements.
    
    Parameters
    ----------
    waterfall: ndarray
        Raw-data shaped as waterfall (nprofiles, nbins).
    ref_prof: int
        Index of reference profile.
    percent: float
        (optional) A number between 0 and 1 describing the percentage of the
        reference profile used to find the baseline.

    Returns
    -------
    baseline: float
        Baseline of reference profile.

    Raises
    ------
    InputError: Exception
        Raised if percentage is not given as a float between 0 and 1.

    '''
    asrt.assert_inrange(percent, 'percent', 0.0, 1.0, expt.InputError,
                        'The chosen percent of raw_data '
                        'to create baseline from is not valid')

    nbins = len(waterfall[ref_prof])
    iend = int(percent * nbins) 

    return np.sum(waterfall[ref_prof, :iend]) / np.floor(percent * nbins)


def rebin(waterfall, rbn, machine=None, dtbin=None):
    '''Rebin waterfall from shape (nprofiles, xbins) to (nprofiles, ybins).
    Based on the rebinning function from the original Fortran program.  
    
    Parameters
    ----------
    waterfall: ndarray
        Raw-data shaped as waterfall (nprofiles, nbins).
    rbn: int
        Rebinning factor. New number of bins will be nbins/rbn.
    machine: Machine
        Object containing machine parameters. If provided, the field
        holding the number of bins in the profile measurement
        will be updated.
    dtbin: float
        Size of profile bins [s]. If provided, the function
        will return the new size of the bins after rebinning.

    Returns
    -------
    rebinned: ndarray
        Rebinned waterfall
    dtbin: float
        (optional) If a dtbin has been provided in the arguments, the
        new size of the profile bins will be returned.
    '''
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
    '''Linear fit to estimate the phase coordinate of the synchronous
    particle. The estimation is done by using the beam reference profile
    set in the `Machine` object.

    Parameters
    ----------
    profiles: Profiles 
        Profiles object containing waterfall and information about the
        measurement.

    Returns
    -------
    fitted_synch_part_x
        X coordinate in the phase space coordinate system of the synchronous
        particle.
    tfoot_low
        Estimation of the lower bunch limit.
        Needed for 'print plot info' function in order to have
        the original output format.
    tfoot_up
        Estimation of the upper bunch limit.
        Needed for 'print plot info' function in order to have
        the original output format.
    
    '''
    ref_idx = profiles.machine.beam_ref_frame
    ref_prof = profiles.waterfall[ref_idx] 
    ref_turn = ref_idx * profiles.machine.dturns

    # Find the upper and lower tangent foor of the bunch.
    # Needed in order to calculate the duration [rad] of the bunch.
    tfoot_up, tfoot_low = _calc_tangentfeet(ref_prof)
    
    # Calculate the duration of the bunch [rad].
    bunch_duration = (tfoot_up - tfoot_low) * profiles.machine.dtbin
    bunch_phaselength = (profiles.machine.h_num * bunch_duration
                         * profiles.machine.omega_rev0[ref_turn])

    # Estimate the synchronous phase.
    x0 = profiles.machine.phi0[ref_turn] - bunch_phaselength / 2.0
    phil = optimize.newton(
            func=physics.phase_low, x0=x0,
            fprime=physics.dphase_low,
            tol=0.0001, maxiter=100,
            args=(profiles.machine, bunch_phaselength, ref_turn))
    
    # Calculates the x coordinate of the synchronous particle given in
    # the phase space coordinate system.
    fitted_synch_part_x = (tfoot_low + (profiles.machine.phi0[ref_turn] - phil)
                           / (profiles.machine.h_num
                           * profiles.machine.omega_rev0[ref_turn]
                           * profiles.machine.dtbin))

    return (fitted_synch_part_x, tfoot_low, tfoot_up)


# Finds foot tangents of profile. Needed to estimate bunch duration
# when performing a fit to find synch_part_x.
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


# Returns index of last bins to the left and right of max valued bin,
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



# returns time, energy and phase space density arrays from a reconstruction,
# requires the homogenous distribution to have been generated by the particles
# class.
def phase_space(tomo, machine, profile = 0):
    
    try:
        machine.dEbin
    except AttributeError:
        raise expt.InputError("""phase_space function requires automatic 
                              phase space generation to have been used""")
    
    density = _make_phase_space(tomo.xp[:,profile], tomo.yp[:,profile], 
                                tomo.weight, machine.nbins)

    tCent = machine.synch_part_x
    ECent = machine.synch_part_y
    
    tRange = (np.arange(machine.nbins) - tCent)*machine.dtbin
    ERange = (np.arange(machine.nbins) - ECent)*machine.dEbin
    
    return tRange, ERange, density
        

# Creates a [nbins, nbins] array populated with the weights of each test 
# particle
def _make_phase_space(xp, yp, weights, nbins):
    
    phaseSpace = np.zeros([nbins, nbins])
    
    for x, y, w in zip(xp, yp, weights):
        phaseSpace[x, y] += w
        
    return phaseSpace
    
    
    
    
    
    
    
    
    

