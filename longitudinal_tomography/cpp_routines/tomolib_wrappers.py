"""Module containing Python wrappers for C++ functions.

Should only be used by advanced users.

:Author(s): **Christoffer Hjertø Grindheim**
"""

import ctypes as ct
import logging
import os
import sys
from glob import glob
from typing import Tuple, TYPE_CHECKING
import tomo.cpp_routines.libtomo as libtomo

import numpy as np

from .. import exceptions as expt

if TYPE_CHECKING:
    from ..tracking.machine import Machine

log = logging.getLogger(__name__)


# =============================================================
# Functions for particle tracking
# =============================================================


def kick(machine: 'Machine', denergy: np.ndarray, dphi: np.ndarray,
         rfv1: np.ndarray, rfv2: np.ndarray, npart: int, turn: int,
         up: bool = True) -> np.ndarray:
    """Wrapper for C++ kick function.

    Particle kick for **one** machine turn.

    Used in the :mod:`longitudinal_tomography.tracking.tracking` module.

    Parameters
    ----------
    machine: Machine
        Object holding machine parameters.
    denergy: ndarray
        1D array holding the energy difference relative to the synchronous
        particle for each particle at a turn.
    dphi: ndarray
        1D array holding the phase difference relative to the synchronous
        particle for each particle at a turn.
    rfv1: ndarray
        1D array holding the radio frequency voltage at RF station 1 for
        each turn, multiplied with the charge state of the particles.
    rfv2: ndarray
        1D array holding the radio frequency voltage at RF station 2 for
        each turn, multiplied with the charge state of the particles.
    npart: int
        The number of tracked particles.
    turn: int
        The current machine turn.
    up: boolean, optional, default=True
        Direction of tracking. Up=True tracks towards last machine turn,
        up=False tracks toward first machine turn.

    Returns
    -------
    denergy: ndarray
        1D array containing the new energy of each particle after voltage kick.
    """
    log.warning('tomo.cpp_routines.tomolib_wrappers.kick has moved to '
                'tomo.cpp_routines.libtomo.kick with the same '
                'function and return signature. This function is '
                'provided for backwards compatibility and can be '
                'removed without further notice.')
    return libtomo.kick(machine, denergy, dphi, rfv1, rfv2,
                        npart, turn, up)


def drift(denergy: np.ndarray, dphi: np.ndarray, drift_coef: np.ndarray,
          npart: int, turn: int, up: bool = True) -> np.ndarray:
    """Wrapper for C++ drift function.

    Particle drift for **one** machine turn

    Used in the :mod:`~longitudinal_tomography.tracking.tracking` module.

    Parameters
    ----------
    denergy: ndarray
        1D array holding the energy difference relative to the synchronous
        particle for each particle at a turn.
    dphi: ndarray
        1D array holding the phase difference relative to the synchronous
        particle for each particle at a turn.
    drift_coef: ndarray
        1D array of drift coefficient at each machine turn.
    npart: int
        The number of tracked particles.
    turn: int
        The current machine turn.
    up: boolean, optional, default=True
        Direction of tracking. Up=True tracks towards last machine turn,
        up=False tracks toward first machine turn.

    Returns
    -------
    dphi: ndarray
        1D array containing the new phase for each particle
        after drifting for a machine turn.
    """
    log.warning('tomo.cpp_routines.tomolib_wrappers.drift has moved to '
                'tomo.cpp_routines.libtomo.drift with the same '
                'function and return signature. This function is '
                'provided for backwards compatibility and can be '
                'removed without further notice.')
    return libtomo.drift(denergy, dphi, drift_coef, npart, turn, up)


def kick_and_drift(xp: np.ndarray, yp: np.ndarray,
                   denergy: np.ndarray, dphi: np.ndarray,
                   rfv1: np.ndarray, rfv2: np.ndarray, rec_prof: int,
                   nturns: int, nparts: int,
                   phi0: np.ndarray = None,
                   deltaE0: np.ndarray = None,
                   omega_rev0: np.ndarray = None,
                   drift_coef: np.ndarray = None,
                   phi12: float = None,
                   h_ratio: float = None,
                   dturns: int = None,
                   machine: 'Machine' = None,
                   ftn_out: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper for full kick and drift algorithm written in C++.

    Tracks all particles from the time frame to be recreated,
    trough all machine turns.

    Used in the :mod:`longitudinal_tomography.tracking.tracking` module.

    Parameters
    ----------
    xp: ndarray
        2D array large enough to hold the phase of each particle
        at every time frame. Shape: (nprofiles, nparts)
    yp: ndarray
        2D array large enough to hold the energy of each particle
        at every time frame. Shape: (nprofiles, nparts)
    denergy: ndarray
        1D array holding the energy difference relative to the synchronous
        particle for each particle the initial turn.
    dphi: ndarray
        1D array holding the phase difference relative to the synchronous
        particle for each particle the initial turn.
    rfv1: ndarray
        Array holding the radio frequency voltage at RF station 1 for each
        turn, multiplied with the charge state of the particles.
    rfv2: ndarray
        Array holding the radio frequency voltage at RF station 2 for each
        turn, multiplied with the charge state of the particles.
    rec_prof: int
        Index of profile to be reconstructed.
    nturns: int
        Total number of machine turns.
    nparts: int
        The number of particles.
    args: tuple
        Arguments can be provided via the args if a machine object is not to
        be used. In this case, the args should be:

        - phi0
        - deltaE0
        - omega_rev0
        - drift_coef
        - phi12
        - h_ratio
        - dturns

        The args will not be used if a Machine object is provided.

    machine: Machine, optional, default=False
        Object containing machine parameters.
ftn_out: boolean, optional, default=False
        Flag to enable printing of status of tracking to stdout.
        The format will be similar to the Fortran version.
        Note that the **information regarding lost particles
        are not valid**.

    Returns
    -------
    xp: ndarray
        2D array holding every particles coordinates in phase [rad]
        at every time frame. Shape: (nprofiles, nparts)
    yp: ndarray
        2D array holding every particles coordinates in energy [eV]
        at every time frame. Shape: (nprofiles, nparts)
    """

    log.warning('tomo.cpp_routines.tomolib_wrappers.kick_and_drift has moved '
                'to tomo.cpp_routines.libtomo.kick_and_drift with the same '
                'function and return signature. This function is '
                'provided for backwards compatibility and can be '
                'removed without further notice.')
    machine_args = [phi0, deltaE0, omega_rev0,
                    drift_coef, phi12, h_ratio, dturns]

    if machine is not None:
        return libtomo.kick_and_drift(xp, yp, denergy, dphi, rfv1, rfv2,
                                      machine, rec_prof, nturns, nparts,
                                      ftn_out)
    elif all([x is not None for x in machine_args]):
        return libtomo.kick_and_drift(xp, yp, denergy, dphi, rfv1, rfv2,
                                      phi0, deltaE0, drift_coef, phi12,
                                      h_ratio, dturns, rec_prof, nturns,
                                      nparts, ftn_out)
    else:
        raise expt.InputError(
            'Wrong input arguments.\n'
            'Either: phi0, deltaE0, omega_rev0, '
            'drift_coef, phi12, h_ratio, dturns '
            'OR machine is required.')


# =============================================================
# Functions for phase space reconstruction
# =============================================================


def back_project(weights: np.ndarray, flat_points: np.ndarray,
                 flat_profiles: np.ndarray, nparts: int, nprofs: int) \
        -> np.ndarray:
    """Wrapper for back projection routine written in C++.
    Used in the :mod:`~longitudinal_tomography.tomography.tomography` module.

    Parameters
    ----------
    weights: ndarray
        1D array containing the weight of each particle.
    flat_points: ndarray
        2D array containing particle coordinates as integers, pointing
        at flattened versions of the waterfall. Shape: (nparts, nprofiles)
    flat_profiles: ndarray
        1D array containing a flattened waterfall.
    nparts: int
        Number of tracked particles.
    nprofs: int
        Number of profiles.

    Returns
    -------
    weights: ndarray
        1D array containing the **new weight** of each particle.
    """
    log.warning('tomo.cpp_routines.tomolib_wrappers.back_project has moved to '
                'tomo.cpp_routines.libtomo.back_project with the same '
                'function and return signature. This function is '
                'provided for backwards compatibility and can be '
                'removed without further notice.')
    return libtomo.back_project(weights, flat_points, flat_profiles, nparts,
                                nprofs)


def project(recreated: np.ndarray, flat_points: np.ndarray,
            weights: np.ndarray,
            nparts: int, nprofs: int, nbins: int) -> np.ndarray:
    """Wrapper projection routine written in C++.
    Used in the :mod:`~longitudinal_tomography.tomography.tomography` module.

    Parameters
    ----------
    recreated: ndarray
        2D array with the shape of the waterfall to be recreated,
        initiated as zero. Shape: (nprofiles, nbins)
    flat_points: ndarray
        2D array containing particle coordinates as integers, pointing
        at flattened versions of the waterfall. Shape: (nparts, nprofiles)
    weights: ndarray
        1D array containing the weight of each particle.
    nparts: int
        Number of tracked particles.
    nprofs: int
        Number of profiles.
    nbins: int
        Number of bins in profiles.

    Returns
    -------
    recreated: ndarray
        2D array containing the projected profiles as waterfall.
        Shape: (nprofiles, nbins)
    """
    log.warning('tomo.cpp_routines.tomolib_wrappers.project has moved to '
                'tomo.cpp_routines.libtomo.project with the same '
                'function and return signature. This function is '
                'provided for backwards compatibility and can be '
                'removed without further notice.')
    return libtomo.project(recreated, flat_points, weights, nparts, nprofs,
                           nbins)


def reconstruct(xp: np.ndarray, waterfall: np.ndarray, niter: int, nbins: int,
                npart: int, nprof: int, verbose: bool):
    """Wrapper for full reconstruction in C++.
    Used in the :mod:`~longitudinal_tomography.tomography.tomography` module.

    Parameters
    ----------
    xp: ndarray
        2D array containing the coordinate of each particle
        at each time frame. Coordinates should given be as integers of
        the phase space coordinate system. shape: (nparts, nprofiles).
    waterfall: ndarray
        2D array containing measured profiles as waterfall.
        Shape: (nprofiles, nbins).
    niter: int
        Number of iterations in the reconstruction process.
    nbins: int
        Number of bins in a profile.
    npart: int
        Number of tracked particles.
    nprof: int
        Number of profiles.
    verbose: boolean
        Flag to indicate that the tomography routine should broadcast its
        status to stdout. The output is identical to the output
        from the Fortran version.

    Returns
    -------
    weights: ndarray
        1D array containing the weight of each particle.
    discr: ndarray
        1D array containing discrepancy at each
        iteration of the reconstruction.
    recreated: ndarray
        2D array containing the projected profiles as waterfall.
        Shape: (nprofiles, nbins)
    """
    log.warning('tomo.cpp_routines.tomolib_wrappers.reconstruct has moved to '
                'tomo.cpp_routines.libtomo.reconstruct with the same '
                'function and return signature. This function is '
                'provided for backwards compatibility and can be '
                'removed without further notice.')
    return libtomo.reconstruct(xp, waterfall, niter, nbins, npart, nprof,
                               verbose)
