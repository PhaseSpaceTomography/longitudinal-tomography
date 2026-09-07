"""Module containing the kick-and-drift algorithm with CuPy.

:Author(s): **Bernardo Abreu Figueiredo**
"""
from __future__ import annotations

import cupy as cp
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cupy.typing import NDArray as CPArray
    from numpy.typing import NDArray as NPArray

log = logging.getLogger(__name__)

def drift_down(dphi: CPArray,
               denergy: CPArray, drift_coef: float,
               n_particles: int) -> CPArray:
    dphi += drift_coef * denergy
    return dphi

def drift_up(dphi: CPArray,
             denergy: CPArray, drift_coef: float,
             n_particles: int) -> CPArray:
    dphi -= drift_coef * denergy
    return dphi

def kick_down(dphi: CPArray,
              denergy: CPArray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> CPArray:
    denergy -= rfv1 * cp.sin(dphi + phi0) \
                      + rfv2 * cp.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick
    return denergy

def kick_up(dphi: CPArray,
            denergy: CPArray, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> CPArray:
    denergy += rfv1 * cp.sin(dphi + phi0) \
                  + rfv2 * cp.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick
    return denergy

def kick_drift_up_simultaneously(dphi: CPArray, denergy: CPArray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> tuple[CPArray, CPArray]:
    dphi -= drift_coef * denergy
    denergy += (rfv1 * cp.sin(dphi + phi0) \
                  + rfv2 * cp.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick)
    return dphi, denergy

def kick_drift_down_simultaneously(dphi: CPArray, denergy: CPArray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> tuple[CPArray, CPArray]:
    denergy -= (rfv1 * cp.sin(dphi + phi0) \
                  + rfv2 * cp.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick)
    dphi += drift_coef * denergy
    return dphi, denergy

def kick_and_drift_cupy(xp: CPArray, yp: CPArray,
                   denergy: CPArray, dphi: CPArray,
                   rfv1: NPArray, rfv2: NPArray,
                   phi0: NPArray,
                   deltaE0: NPArray,
                   drift_coef: NPArray,
                   phi12: float,
                   h_ratio: float,
                   dturns: int,
                   rec_prof: int,
                   deltaturn: int,
                   nturns: int,
                   nparts: int,
                   fortran_flag,
                   callback) -> tuple[CPArray, CPArray]:
    
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