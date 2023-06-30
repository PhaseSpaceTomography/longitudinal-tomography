"""Module containing the kick-and-drift algorithm derived from the cpp functions
with Numba optimizations.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
from typing import Tuple
import logging
from numba import njit
import math
from ..utils.execution_mode import Mode

log = logging.getLogger(__name__)

@njit(parallel=True)
def drift(denergy: np.ndarray,
          dphi: np.ndarray,
          drift_coef: np.ndarray, npart: int, turn: int,
          up: bool) -> np.ndarray:
    if up:
        return drift_up(dphi, denergy, drift_coef[turn], npart)
    else:
        return drift_down(dphi, denergy, drift_coef[turn], npart)

@njit(parallel=True)
def drift_down(dphi: np.ndarray,
               denergy: np.ndarray, drift_coef: float,
               n_particles: int) -> np.ndarray:
    dphi += drift_coef * denergy
    return dphi

@njit(parallel=True)
def drift_up(dphi: np.ndarray,
             denergy: np.ndarray, drift_coef: float,
             n_particles: int) -> np.ndarray:
    dphi -= drift_coef * denergy
    return dphi

@njit(parallel=True)
def kick_drift_up_simultaneously(dphi: np.ndarray, denergy: np.ndarray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> Tuple[np.ndarray, np.ndarray]:
    dphi -= drift_coef * denergy
    denergy += (rfv1 * np.sin(dphi + phi0) \
                      + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick)
    return dphi, denergy

@njit(parallel=True)
def kick_drift_down_simultaneously(dphi: np.ndarray, denergy: np.ndarray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> Tuple[np.ndarray, np.ndarray]:
    denergy -= (rfv1 * np.sin(dphi + phi0) \
                      + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick)
    dphi += drift_coef * denergy
    return dphi, denergy

@njit(parallel=True)
def kick(machine: object, denergy: np.ndarray,
         dphi: np.ndarray,
         rfv1: np.ndarray,
         rfv2: np.ndarray, npart: int, turn: int,
         up: bool) -> np.ndarray:

    if up:
        return kick_up(dphi, denergy, rfv1[turn], rfv2[turn],
                        machine.phi0[turn], machine.phi12, machine.h_ratio, npart,
                        machine.deltaE0[turn])
    else:
        return kick_down(dphi, denergy, rfv1[turn], rfv2[turn],
                        machine.phi0[turn], machine.phi12, machine.h_ratio, npart,
                        machine.deltaE0[turn])

def kick_and_drift_numba(xp: np.ndarray, yp: np.ndarray,
                   denergy: np.ndarray, dphi: np.ndarray,
                   rfv1: np.ndarray, rfv2: np.ndarray, rec_prof: int,
                   nturns: int, nparts: int,
                   phi0: np.ndarray,
                   deltaE0: np.ndarray,
                   drift_coef: np.ndarray,
                   phi12: float,
                   h_ratio: float,
                   dturns: int,
                   deltaturn: int,
                   machine: 'Machine',
                   ftn_out: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    # PREPARATION: see libtomo.kick_and_drift_machine/scalar/array
    phi12_arr = np.full(nturns+1, phi12)
    # Preparation end

    profile = rec_prof
    turn = rec_prof * dturns + deltaturn

    if deltaturn < 0:
        profile -= 1

    # Value-based copy to avoid side-effects
    xp[profile] = np.copy(dphi)
    yp[profile] = np.copy(denergy)

    together = False

    dphi2 = np.copy(dphi)
    denergy2 = np.copy(denergy)

    while turn < nturns:
        if together:
            turn += 1
            dphi, denergy = kick_drift_up_simultaneously(dphi, denergy, drift_coef[turn-1], rfv1[turn], rfv2[turn], phi0[turn],
                                                      phi12_arr[turn], h_ratio, nparts, deltaE0[turn])
        else:
            dphi = drift_up(dphi, denergy, drift_coef[turn], nparts)

            turn += 1
            denergy = kick_up(dphi, denergy, rfv1[turn], rfv2[turn], phi0[turn], phi12_arr[turn],
                h_ratio, nparts, deltaE0[turn])

        if turn % dturns == 0:
            profile += 1
            xp[profile] = np.copy(dphi)
            yp[profile] = np.copy(denergy)

            if ftn_out:
                log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                        0.000% went outside the image width.")

    profile = rec_prof
    turn = rec_prof * dturns

    if profile > 0:
        # going back to initial coordinates
        dphi = np.copy(xp[rec_prof])
        denergy = np.copy(yp[rec_prof])

        # Downwards
        while turn > 0:
            if together:
                dphi, denergy = kick_drift_down_simultaneously(dphi, denergy, drift_coef[turn-1], rfv1[turn], rfv2[turn],
                                                                    phi0[turn], phi12_arr[turn], h_ratio, nparts, deltaE0[turn])
                turn -= 1
            else:
                denergy = kick_down(dphi, denergy, rfv1[turn], rfv2[turn], phi0[turn],
                        phi12_arr[turn], h_ratio, nparts, deltaE0[turn])
                turn -= 1
                dphi = drift_down(dphi, denergy, drift_coef[turn], nparts)

            if (turn % dturns == 0):
                profile -= 1
                xp[profile] = np.copy(dphi)
                yp[profile] = np.copy(denergy)

                if ftn_out:
                    log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                                0.000% went outside the image width.")
    return xp, yp

@njit(parallel=True)
def kick_down(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> np.ndarray:
    denergy -= rfv1 * np.sin(dphi + phi0) \
                      + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick
    return denergy

@njit(parallel=True)
def kick_up(dphi: np.ndarray,
            denergy: np.ndarray, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> np.ndarray:
    denergy += rfv1 * np.sin(dphi + phi0) \
                  + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick
    return denergy