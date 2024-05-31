"""Module containing the kick-and-drift algorithm with CuPy.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
import cupy as cp
import logging
from typing import Tuple

log = logging.getLogger(__name__)

def drift_down(dphi: cp.ndarray,
               denergy: cp.ndarray, drift_coef: float,
               n_particles: int) -> cp.ndarray:
    dphi += drift_coef * denergy
    return dphi

def drift_up(dphi: cp.ndarray,
             denergy: cp.ndarray, drift_coef: float,
             n_particles: int) -> cp.ndarray:
    dphi -= drift_coef * denergy
    return dphi

def kick_down(dphi: cp.ndarray,
              denergy: cp.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> cp.ndarray:
    denergy -= rfv1 * cp.sin(dphi + phi0) \
                      + rfv2 * cp.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick
    return denergy

def kick_up(dphi: cp.ndarray,
            denergy: cp.ndarray, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> cp.ndarray:
    denergy += rfv1 * cp.sin(dphi + phi0) \
                  + rfv2 * cp.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick
    return denergy

def kick_drift_up_simultaneously(dphi: cp.ndarray, denergy: cp.ndarray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> Tuple[cp.ndarray, cp.ndarray]:
    dphi -= drift_coef * denergy
    denergy += (rfv1 * cp.sin(dphi + phi0) \
                  + rfv2 * cp.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick)
    return dphi, denergy

def kick_drift_down_simultaneously(dphi: cp.ndarray, denergy: cp.ndarray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> Tuple[cp.ndarray, cp.ndarray]:
    denergy -= (rfv1 * cp.sin(dphi + phi0) \
                  + rfv2 * cp.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick)
    dphi += drift_coef * denergy
    return dphi, denergy

def kick_and_drift_cupy(xp: cp.ndarray, yp: cp.ndarray,
                   denergy: cp.ndarray, dphi: cp.ndarray,
                   rfv1: np.ndarray, rfv2: np.ndarray,
                   phi0: np.ndarray,
                   deltaE0: np.ndarray,
                   drift_coef: np.ndarray,
                   phi12: float,
                   h_ratio: float,
                   dturns: int,
                   rec_prof: int,
                   deltaturn: int,
                   nturns: int,
                   nparts: int,
                   fortran_flag,
                   callback) -> Tuple[cp.ndarray, cp.ndarray]:
    
    drift_coef = cp.asarray(drift_coef)
    phi0 = cp.asarray(phi0)
    deltaE0 = cp.asarray(deltaE0)
    rfv1 = cp.asarray(rfv1)
    rfv2 = cp.asarray(rfv2)

    phi12_arr = cp.full(nturns+1, phi12)
    # Preparation end

    profile = rec_prof
    turn = rec_prof * dturns + deltaturn

    if deltaturn < 0:
        profile -= 1

    # Value-based copy to avoid side-effects
    xp[profile] = cp.copy(dphi)
    yp[profile] = cp.copy(denergy)

    while turn < nturns:
        turn += 1
        dphi, denergy = kick_drift_up_simultaneously(dphi, denergy, drift_coef[turn-1], rfv1[turn], rfv2[turn],
                                                        phi0[turn], phi12_arr[turn], h_ratio, nparts, deltaE0[turn])

        if turn % dturns == 0:
            profile += 1

            xp[profile] = cp.copy(dphi)
            yp[profile] = cp.copy(denergy)

    profile = rec_prof
    turn = rec_prof * dturns

    if profile > 0:
        # going back to initial coordinates
        dphi = cp.copy(xp[rec_prof])
        denergy = cp.copy(yp[rec_prof])

        # Downwards
        while turn > 0:
            dphi, denergy = kick_drift_down_simultaneously(dphi, denergy, drift_coef[turn-1], rfv1[turn], rfv2[turn],
                                                                    phi0[turn], phi12_arr[turn], h_ratio, nparts, deltaE0[turn])
            turn -= 1

            if (turn % dturns == 0):
                profile -= 1
                xp[profile] = cp.copy(dphi)
                yp[profile] = cp.copy(denergy)