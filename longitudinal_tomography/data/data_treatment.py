"""Module containing functions for treatment of data.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
from __future__ import annotations
from typing import Tuple, TYPE_CHECKING, Union
from warnings import warn

import numpy as np
import itertools as itl

from .. import exceptions as expt
from ..utils import physics
from ..cpp_routines import libtomo
from . import pre_process

import logging

if TYPE_CHECKING:
    from .profiles import Profiles
    from ..tracking.machine import Machine
    from ..tracking.machine_base import MachineABC
    from ..tomography.__tomography import TomographyABC

__all__ = ['rebin', 'fit_synch_part_x', 'phase_space']

log = logging.getLogger(__name__)


def rebin(waterfall: np.ndarray, rbn: int, dtbin: float = None,
          synch_part_x: float = None) \
        -> Union[Tuple[np.ndarray, float, float], Tuple[np.ndarray, float]]:
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
    # warn('The rebin function has been moved to '
    #      'longitudinal_tomography.data.pre_process')
    return pre_process.rebin(waterfall, rbn, dtbin, synch_part_x)


# Original function for finding synch_part_x
# Finds synch_part_x based on a linear fit on a reference profile.
def fit_synch_part_x(profiles: 'Profiles') -> Tuple[np.ndarray, float, float]:
    """Linear fit to estimate the phase coordinate of the synchronous
    particle. The found phase is returned as a x-coordinate of the phase space
    coordinate systems in fractions of bins. The estimation is done at
    the beam reference profile, which is set in the
    :class:`longitudinal_tomography.tracking.machine.Machine` object.

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
        Needed for
        :func:`longitudinal_tomography.utils.tomo_output.write_plotinfo_ftn`
        function in order to write the original output format.
    upper bunch limit
        Estimation of the upper bunch limit in bin numbers.
        Needed for
        :func:`longitudinal_tomography.utils.tomo_output.write_plotinfo_ftn`
        function in order to write the original output format.

    """
    # warn('The fit_synch_part_x function has moved to '
    #      'longitudinal_tomography.data.pre_process')
    return pre_process.fit_synch_part_x(profiles)


def phase_space(tomo: 'TomographyABC', machine: 'MachineABC',
                reconstr_idx: int = 0) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """returns time, energy and phase space density arrays from a
    reconstruction, requires the homogenous distribution to have been
    generated by the particles class.

    Parameters
    ----------
    tomo: TomographyABC
        Object holding the information about a tomographic reconstruction.
    machine: Machine
        Object holding information about machine and reconstruction parameters.
    reconstr_idx: int
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
    if machine.dEbin is None:
        raise expt.InputError("""phase_space function requires automatic
                              phase space generation to have been used""")

    return phase_space_from_coordinates(tomo.xp[:, reconstr_idx],
                                        tomo.yp[:, reconstr_idx],
                                        tomo.weight, machine.nbins,
                                        machine.synch_part_x,
                                        machine.synch_part_y, machine.dtbin,
                                        machine.dEbin)


def phase_space_from_coordinates(xp: np.ndarray, yp: np.ndarray,
                                 weight: np.ndarray, nbins: int,
                                 synch_x: float, synch_y: float,
                                 dtbin: float, dEbin: float) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns time, energy and phase space density arrays from a
    reconstruction.

    Parameters
    ----------
    xp : np.ndarray
        An array of x-coordinates in bin numbers for the reconstructed phase
        space.
    yp : np.ndarray
        An array of y-coordinates in bin numbers for the reconstructed phase
        space.
    weight : np.ndarray
        The array of particle weights for the reconstructed phase space.
    nbins : int
        Number of bins in each dimension.
    synch_x : float
        The bin number in time of the synchronous particle.
    synch_y : float
        The bin number in energy of the synchronous particle.
    dtbin : float
        The number of s per time bin.
    dEbin : float
        The number of eV per energy bin.

    Returns
    -------
    t_range : np.ndarray
        1D array of the center times for each time bin.
    E_range : np.ndarray
        1D array of the center energies for each energy bin.
    density : np.ndarray
        2D array of bin weights in phase space.
    """

    density = libtomo.make_phase_space(xp, yp, weight, nbins)

    t_range = (np.arange(nbins) - synch_x) * dtbin
    E_range = (np.arange(nbins) - synch_y) * dEbin

    return t_range, E_range, density


# Creates a [nbins, nbins] array populated with the weights of each test
# particle
def _make_phase_space(xp: np.ndarray, yp: np.ndarray, weights: np.ndarray,
                      nbins: int) -> np.ndarray:
    log.warning('tomo.data.data_treatment._make_phase_space has moved to '
                'the C++ library at '
                'tomo.cpp_routines.libtomo.make_phase_space. '
                'This function is provided for backwards compatibility '
                'and can be removed without further notice.')
    return libtomo.make_phase_space(xp, yp, weights, nbins)


def calc_baseline_ftn(*args):
    # warn('This function has moved to '
    #      'longitudinal_tomography.compat.fortran.')
    from longitudinal_tomography.compat.fortran import calc_baseline
    return calc_baseline(*args)


def density_to_macro(tRange: np.ndarray, ERange: np.ndarray,
                     density: np.ndarray, n_macro: int,
                     threshold: float = 1E-5) -> Tuple:
    """

    Takes a time range, energy range and density function and converts it to
    n_macro number macroparticles to reproduce the distribution via randomly
    sampling from it.


    Parameters
    ----------
    t_range : np.ndarray
        1D array of the center times for each time bin.
    E_range : np.ndarray
        1D array of the center energies for each energy bin.
    density : np.ndarray
        2D array of bin weights in phase space.
    n_macro: int
        Amount of macroparticles to be generated
    threshold: float
        Threshhold under which density is set to 0.

    Returns
    -------
    dt: np.ndarray
        1D array of time coordinates of each macroparticle
    dE: np.ndarray
        1D array of energy coordinates of each macroparticle

    """

    dtbin = tRange[1] - tRange[0]
    dEbin = ERange[1] - ERange[0]

    sumDens = np.sum(density)
    density[density / sumDens < threshold] = 0

    prob = density.flatten()
    prob = prob / np.sum(prob)
    coords = np.array(list(itl.product(tRange, ERange)))
    n_macro = int(n_macro)

    samples = np.random.choice(prob.shape[0], n_macro, p=prob)

    dt = coords[samples, 0] + dtbin * (np.random.random(n_macro) - 0.5)
    dE = coords[samples, 1] + dEbin * (np.random.random(n_macro) - 0.5)

    return dt, dE
