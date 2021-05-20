"""Module containing phase space post-processing functions.

:Author(s): **Anton Lu**
"""
from numbers import Number
from typing import Union, Dict

import numpy as np
from scipy import constants as cont
from multipledispatch import dispatch
from .. import assertions as asrt

__all__ = ['post_process', 'rms_dpp', 'emittance_rms',
           'emittance_90', 'emittance_fractional']

m_p = cont.value('proton mass energy equivalent in MeV') * 1e6


def post_process(phase_space: np.ndarray, t_bins: np.ndarray,
                 e_bins: np.ndarray, energy: Number, mass: Number = m_p) \
        -> Dict[str, Union[float, np.ndarray]]:
    """
    Convenience function that provides an all-on-one post-processing method.

    Calculates the RMS emittance, 90% emittance and RMS dp/p of the given
    phase space

    Parameters
    ----------
    phase_space : np.ndarray
        A 2-dimensional array that represents the phase space of size
        (#t_bins, #e_bins)
    t_bins : np.ndarray
        A 1 or 2-dimensional vector that represent the bins on the time axis
    e_bins : np.ndarray
        A 1 or 2-dimensional vector that represents the bins on the energy axis
    energy : Number
        The energy of the beam at the reconstructed profile
    mass : Number
        The rest mass of the particle(s)

    Returns
    -------
    dict :
        A dictionary with the keys
        - emittance_rms
        - emittance_90
        - rms_dp/p
        Where k/v pair is one beam quality measure

    """

    time_proj = phase_space.sum(1)
    energy_proj = phase_space.sum(0)
    t_std = _std_from_histogram(time_proj, t_bins)
    e_std = _std_from_histogram(energy_proj, e_bins)

    out = {
        'emittance_rms': emittance_rms(t_std, e_std),
        'emittance_90': emittance_90(phase_space, t_bins, e_bins),
        'rms_dp/p': rms_dpp(e_std, energy, mass)
    }

    return out


@dispatch(np.ndarray, np.ndarray, np.ndarray)
def emittance_rms(histogram: np.ndarray,
                  t_bins: np.ndarray,
                  e_bins: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculates the RMS emittance for the given phase space
    by first calculating the standard deviation of the time and energy
    histograms

    Parameters
    ----------
    histogram : np.ndarray
        A 2-dimensional array that represents the phase space of size
        (#t_bins, #e_bins)
    t_bins : np.ndarray
        A 1 or 2-dimensional vector that represent the bins on the time axis
    e_bins : np.ndarray
        A 1 or 2-dimensional vector that represents the bins on the energy axis

    Returns
    -------
    float :
        The calculated RMS emittance
    """
    t_std = _std_from_histogram(histogram.sum(axis=1), t_bins)
    e_std = _std_from_histogram(histogram.sum(axis=0), e_bins)

    return emittance_rms(t_std, e_std)


@dispatch(float, float)
def emittance_rms(sigma_t: Union[float, np.ndarray],
                  sigma_e: Union[float, np.ndarray]) \
        -> Union[float, np.ndarray]:
    """
    Calculates the RMS emittance from the standard deviation of
    time and energy.

    Parameters
    ----------
    sigma_t : float
        Standard deviation of the time axis
    sigma_e : float
        Standard deviation of the energy axis

    Returns
    -------
    float :
        The calculated RMS emittance
    """
    return np.pi * sigma_t * sigma_e


def emittance_90(phase_space: np.ndarray,
                 t_bins: np.ndarray,
                 e_bins: np.ndarray) -> float:
    """
    Convenience function to calculate the 90% emittance that calls
    the emittance_fractional function.

    Parameters
    ----------
    phase_space : np.ndarray
        A 2-dimensional array that represents the phase space
    t_bins : np.ndarray
        A 1 or 2-dimensional vector that represent the bins on the time axis
    e_bins : np.ndarray
        A 1 or 2-dimensional vector that represents the bins on the energy axis

    Returns
    -------
    float :
        The 90% emittance of the passed phase space

    """
    return emittance_fractional(phase_space, t_bins, e_bins, fraction=90)


@dispatch(np.ndarray, np.ndarray, np.ndarray, fraction=float)
def emittance_fractional(histogram: np.ndarray,
                         t_bins: np.ndarray,
                         e_bins: np.ndarray,
                         fraction: float = 90) -> float:
    """
    Calculates the fractional emittance of a given phase space (histogram),
    defaults to 90% emittance if the `fraction` argument is left out.

    Parameters
    ----------
    histogram : np.ndarray
        A 2-dimensional array that represents the phase space
    t_bins : np.ndarray
        A 1 or 2-dimensional vector that represent the bins on the time axis
    e_bins : np.ndarray
        A 1 or 2-dimensional vector that represents the bins on the energy axis
    fraction: float
        A number between 0 and 100 (not inclusive), calculates the fraction%
        emittance.

    Returns
    -------
    float :
        The fractional emittance of the passed phase space

    """
    # TODO: explore options for faster calculations
    asrt.assert_less_or_equal(fraction, 'fraction', 100, ValueError)
    asrt.assert_greater_or_equal(fraction, 'fraction', 0, ValueError)

    fraction /= 100

    histogram = histogram / histogram.sum()
    histogram = histogram.ravel()
    histogram[::-1].sort()
    cumsum = histogram.cumsum()
    n_bins = np.argmax(cumsum >= fraction * histogram.sum())

    t_bin_width = t_bins[1] - t_bins[0]
    e_bin_width = e_bins[1] - e_bins[0]

    return n_bins * (t_bin_width * e_bin_width)


@dispatch(Number, Number, Number)
def rms_dpp(energy_std: Number, energy: Number, mass: Number) -> float:
    """
    Calculates the RMS dp/p using the energy projection histogram.
    Provided as a convenience function that first calculates
    the standard deviation of the energy and then calculates the
    RMS dp/p using the std.

    dp is calculated using the relation ((p + dp)c)^2 + m^2c^4 = (E + dE)^2

    Parameters
    ----------
    energy_std : Number
        The standard deviation of the energy axis
    energy : Number
        The energy of the beam at the reconstructed profile
    momentum : Number
        The momentum of the beam at the reconstructed profile
    mass : Number
        The rest mass of the particle(s)

    Returns
    -------
    float :
        The RMS dp/p of particle beam
    """
    momentum = np.sqrt(energy ** 2 - mass ** 2)
    dp = np.sqrt(np.power(energy + energy_std, 2) - mass ** 2) - momentum

    rms_dpp = dp / momentum

    return rms_dpp


@dispatch(np.ndarray, np.ndarray, Number, Number)
def rms_dpp(energy_proj: np.ndarray, energy_bins: np.ndarray,
            energy: Number,
            mass: Number) -> float:
    """
    Calculates the RMS dp/p using the energy projection histogram.
    Provided as a convenience function that first calculates
    the standard deviation of the energy and then calculates the
    RMS dp/p using the std.

    Parameters
    ----------
    energy_proj : np.ndarray
        A 1 or 2-dimensional vector that represents the energy projection of
        the phase space
    energy_bins : np.ndarray
        A 1 or 2-dimensional vector that represents the bins on the energy axis
    energy : Number
        The energy of the beam at the reconstructed profile
    momentum : Number
        The momentum of the beam at the reconstructed profile
    mass : Number
        The rest mass of the particle(s)

    Returns
    -------
    float :
        The RMS dp/p of the energy projection
    """
    energy_std = _std_from_histogram(energy_proj, energy_bins)

    return rms_dpp(energy_std, energy, mass)


def _std_from_histogram(histogram: np.ndarray, bins: np.ndarray) -> float:
    """
    Calculates the standard deviation of a given histogram and the bin values

    Parameters
    ----------
    histogram : np.ndarray
        A histogram to calculate the standard deviation for
    bins : np.ndarray
        The bins of the histogram. Dimensions must match that of the histogram

    Returns
    -------
    float :
        The standard deviation of the histogram
    """
    asrt.assert_array_numel_equal((histogram, bins), ('histogram', 'bins'),
                                  histogram.size)

    # guard against 2 dim arrays
    bins = bins.flatten()
    histogram = histogram.flatten()

    bin_width = bins[1] - bins[0]
    mid_values = bins + bin_width / 2

    mean = mid_values * histogram / histogram.sum()
    var = np.sum(histogram * np.power(mid_values - mean, 2)) / histogram.sum()

    std = np.sqrt(var)
    return std
