import numpy as np

# Convert from one-dimentional list of raw data to waterfall.
# Works on a copy of the raw data
def raw_data_to_waterfall(machine, raw_data):
    waterfall = np.copy(raw_data)
    waterfall = raw_data.reshape((machine.nprofiles, machine.framelength))

    if machine.postskip_length > 0:
        waterfall = waterfall[:, machine.preskip_length:
                                -machine.postskip_length]
    else:
        waterfall = waterfall[:, machine.preskip_length:]

    return waterfall


# Original function for subtracting baseline of raw data input profiles.
# Finds the baseline from the first 5% (by default)
#  of the beam reference profile.
def calc_baseline_ftn(waterfall, ref_prof, percent=0.05):
    assert_inrange(percent, 'percent', 0.0, 1.0, InputError,
                   'The chosen percent of raw_data '
                   'to create baseline from is not valid')

    nbins = len(waterfall[ref_prof])
    iend = int(percent * nbins) 

    return np.sum(waterfall[ref_prof, :iend]) / np.floor(percent * nbins)


def rebin(waterfall, rbn):
    data = np.copy(waterfall)

    # Check that there is enough data to for the given rebin factor.

    if data.shape[1] % rbn == 0:
        rebinned = _rebin_even(data, rbn)
    else:
        rebinned = _rebin_odd(data, rbn)

    return rebinned

# Rebins an 2d array given a rebin factor (rbn).
# The given array MUST have a length equal to an even number.
def _rebin_even(data, rbn):
    if data.shape[1] % rbn != 0:
        raise AssertionError('Length of input data must be an even number.')

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
def _rebin_odd(data, rbn):
    nprofs = data.shape[0]
    nbins = data.shape[1]
    ans = np.zeros((nprofs, int(nbins / rbn) + 1))
    ans[:,:-1] = _rebin_even(data[:,:-1], rbn)
    ans[:,-1] = _rebin_last(data, rbn)[:, 0]
    return ans


# Rebins last indices of an 2d array given a rebin factor (rbn).
# Needed for the rebinning of odd arrays.
def _rebin_last(data, rbn):
    nprofs = data.shape[0]
    nbins = data.shape[1]
    new_nbins = int(nbins / rbn)

    i0 = (new_nbins - 1) * rbn
    ans = np.copy(data[:,i0:])
    ans = np.sum(ans, axis=1)
    ans[:] *= rbn / (nbins - (new_nbins - 1) * rbn)
    ans = ans.reshape((nprofs, 1))
    return ans
