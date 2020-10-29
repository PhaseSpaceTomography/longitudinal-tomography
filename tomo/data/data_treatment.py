"""Module containing functions for treatment of data.

:Author(s): **Christoffer Hjertø Grindheim**
"""
from typing import Tuple, TYPE_CHECKING

import numpy as np
from scipy import optimize

from tomo import exceptions as expt
from tomo.utils import physics

if TYPE_CHECKING:
    from tomo.data.profiles import Profiles
    from tomo.tracking.machine import Machine
    from tomo.tomography.__tomography import Tomography


def rebin(waterfall: np.ndarray, rbn: int, dtbin: float = None,
          synch_part_x: float = None) \
        -> Tuple[np.ndarray, float, float]:
    """Rebin waterfall from shape (P, X) to (P, Y).
    P is the number of profiles, X is the original number of bins,
    and Y is the number of bins after the re-binning.

    The algorithm is based on the rebinning function from the
    original tomography program.

    An array of length N, rebinned with a rebin factor of R will result
    in a array of length N / R. If N is not dividable on R, the resulting
    array will have the length (N / R) + 1.

    Parameters
    ----------
    waterfall: ndarray
        2D array of raw-data shaped as waterfall. Shape: (nprofiles, nbins).
    rbn: int
        Rebinning factor.
    dtbin: float
        Size of profile bins [s]. If provided, the function
        will return the new size of the bins after rebinning.
    synch_part_x: float
        x-coordinate of synchronous particle, measured in bins.
        If provided, the function will return its updated coordinate.

    Returns
    -------
    rebinned: ndarray
        Rebinned waterfall
    dtbin: float, optional, default=None
        If a dtbin has been provided in the arguments, the
        new size of the profile bins will be returned. Otherwise,
        None will be returned.
    synch_part_x: float, optional, default=None
        If a synch_part_x has been provided in the arguments, the
        new x-coordinate of the synchronous particle in bins will be returned.
        Otherwise, None will be returned.
    """
    data = np.copy(waterfall)

    # Check that there is enough data to for the given rebin factor.
    if data.shape[1] % rbn == 0:
        rebinned = _rebin_dividable(data, rbn)
    else:
        rebinned = _rebin_individable(data, rbn)

    if dtbin is not None:
        dtbin *= rbn
    if synch_part_x is not None:
        synch_part_x /= rbn

    return rebinned, dtbin, synch_part_x


# Rebins an 2d array given a rebin factor (rbn).
# The given array MUST have a length equal to an even number.
def _rebin_dividable(data: np.ndarray, rbn: int) -> np.ndarray:
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
def _rebin_individable(data: np.ndarray, rbn: int) -> np.ndarray:
    nprofs = data.shape[0]
    nbins = data.shape[1]

    ans = np.zeros((nprofs, int(nbins / rbn) + 1))

    last_data_idx = int(nbins / rbn) * rbn
    ans[:, :-1] = _rebin_dividable(data[:, :last_data_idx], rbn)
    ans[:, -1] = _rebin_last(data, rbn)[:, 0]
    return ans


# Rebins last indices of an 2d array given a rebin factor (rbn).
# Needed for the rebinning of odd arrays.
def _rebin_last(data: np.ndarray, rbn: int) -> np.ndarray:
    nprofs = data.shape[0]
    nbins = data.shape[1]

    i0 = (int(nbins / rbn) - 1) * rbn
    ans = np.copy(data[:, i0:])
    ans = np.sum(ans, axis=1)
    ans[:] *= rbn / (nbins - i0)
    ans = ans.reshape((nprofs, 1))
    return ans


# Original function for finding synch_part_x
# Finds synch_part_x based on a linear fit on a reference profile.
def fit_synch_part_x(profiles: 'Profiles') -> Tuple[np.ndarray, float, float]:
    """Linear fit to estimate the phase coordinate of the synchronous
    particle. The found phase is returned as a x-coordinate of the phase space
    coordinate systems in fractions of bins. The estimation is done at
    the beam reference profile, which is set in the
    :class:`tomo.tracking.machine.Machine` object.

    Parameters
    ----------
    profiles: Profiles
        Profiles object containing waterfall and information about the
        measurement.

    Returns
    -------
    fitted_synch_part_x
        X coordinate in the phase space coordinate system of the synchronous
        particle given in bin numbers.
    lower bunch limit
        Estimation of the lower bunch limit in bin numbers.
        Needed for :func:`tomo.utils.tomo_output.write_plotinfo_ftn`
        function in order to write the original output format.
    upper bunch limit
        Estimation of the upper bunch limit in bin numbers.
        Needed for :func:`tomo.utils.tomo_output.write_plotinfo_ftn`
        function in order to write the original output format.

    """
    ref_idx = profiles.machine.beam_ref_frame
    ref_prof = profiles.waterfall[ref_idx]
    ref_turn = ref_idx * profiles.machine.dturns

    # Find the upper and lower tangent for of the bunch.
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

    return fitted_synch_part_x, tfoot_low, tfoot_up


# Finds foot tangents of profile. Needed to estimate bunch duration
# when performing a fit to find synch_part_x.
def _calc_tangentfeet(ref_prof: np.ndarray) -> Tuple[float, float]:
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
def _calc_tangentbins(ref_profile: np.ndarray, nbins: int,
                      threshold_coeff: float = 0.15) -> Tuple[float, float]:
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


def phase_space(tomo: 'Tomography', machine: 'Machine', profile: int = 0) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """returns time, energy and phase space density arrays from a
    reconstruction, requires the homogenous distribution to have been
    generated by the particles class.

    Parameters
    ----------
    tomo: Tomography
        Object holding the information about a tomographic reconstruction.
    machine: Machine
        Object holding information about machine and reconstruction parameters.
    profile: int
        Index of profile to be reconstructed.

    Returns
    -------
    t_range: ndarray
        1D array containing time axis of reconstructed phase space image.
    E_range: ndarray
        1D array containing energy axis of reconstructed phase space image.
    density: ndarray
        2D array containing the reconstructed phase space image.

    Raises
    ------
    InputError: Exception
        phase_space function requires automatic phase space generation
        to have been used.

    """
    try:
        machine.dEbin
    except AttributeError:
        raise expt.InputError("""phase_space function requires automatic
                              phase space generation to have been used""")

    density = _make_phase_space(tomo.xp[:, profile], tomo.yp[:, profile],
                                tomo.weight, machine.nbins)

    t_cent = machine.synch_part_x
    E_cent = machine.synch_part_y

    t_range = (np.arange(machine.nbins) - t_cent) * machine.dtbin
    E_range = (np.arange(machine.nbins) - E_cent) * machine.dEbin

    return t_range, E_range, density


# Creates a [nbins, nbins] array populated with the weights of each test
# particle
def _make_phase_space(xp: np.ndarray, yp: np.ndarray, weights: np.ndarray,
                      nbins: int) -> np.ndarray:
    phase_space = np.zeros([nbins, nbins])

    for x, y, w in zip(xp, yp, weights):
        phase_space[x, y] += w

    return phase_space
