"""Module containing the kick-and-drift algorithm derived from the cpp functions.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
from typing import Tuple
import logging
from numba import njit, vectorize, prange
import math
from ..cpp_routines import libtomo
from ..utils.execution_mode import Mode

log = logging.getLogger(__name__)

UP = -1
DOWN = 1

def drift(denergy: np.ndarray,
          dphi: np.ndarray,
          drift_coef: np.ndarray, npart: int, turn: int,
          up: bool = ...) -> np.ndarray:
    if up:
        return drift_up(dphi, denergy, drift_coef[turn], npart)
    else:
        return drift_down(dphi, denergy, drift_coef[turn], npart)

def drift_down(dphi: np.ndarray,
               denergy: np.ndarray, drift_coef: float,
               n_particles: int) -> np.ndarray:
    dphi += drift_coef * denergy
    return dphi

def drift_down_unrolled(dphi: np.ndarray,
                        denergy: np.ndarray, drift_coef: float,
                        n_particles: int) -> np.ndarray:
    for i in range(n_particles):
        dphi[i] += drift_coef * denergy[i]
    return dphi

def drift_down_unrolled_parallel(dphi: np.ndarray,
                                 denergy: np.ndarray, drift_coef: float,
                                 n_particles: int) -> np.ndarray:
    for i in prange(n_particles):
        dphi[i] += drift_coef * denergy[i]
    return dphi

def drift_down_vectorized(dphi: float, denergy: float, drift_coef: float, n_particles: int) -> float:
    return dphi + drift_coef * denergy
    dphi += drift_coef * denergy

def drift_down_cpp(dphi: np.ndarray,
                   denergy: np.ndarray, drift_coef: float,
                   n_particles: int) -> np.ndarray:
    libtomo.drift_down(dphi, denergy, drift_coef, n_particles)
    return dphi


def drift_up(dphi: np.ndarray,
             denergy: np.ndarray, drift_coef: float,
             n_particles: int) -> np.ndarray:
    dphi -= drift_coef * denergy
    return dphi

def drift_up_unrolled(dphi: np.ndarray,
                      denergy: np.ndarray, drift_coef: float,
                      n_particles: int) -> np.ndarray:
    for i in range(n_particles):
        dphi[i] -= drift_coef * denergy[i]
    return dphi

def drift_up_unrolled_parallel(dphi: np.ndarray,
                      denergy: np.ndarray, drift_coef: float,
                      n_particles: int) -> np.ndarray:
    for i in prange(n_particles):
        dphi[i] -= drift_coef * denergy[i]
    return dphi

def drift_up_vectorized(dphi: float, denergy: float, drift_coef: float, n_particles: int) -> float:
    return dphi - drift_coef * denergy

def drift_up_cpp(dphi: np.ndarray,
                   denergy: np.ndarray, drift_coef: float,
                   n_particles: int) -> np.ndarray:
    libtomo.drift_up(dphi, denergy, drift_coef, n_particles)
    return dphi

def kick_drift_up_simultaneously(dphi: np.ndarray, denergy: np.ndarray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> Tuple[np.ndarray, np.ndarray]:
    dphi -= drift_coef * denergy
    denergy += (rfv1 * np.sin(dphi + phi0) \
                      + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick)
    return dphi, denergy

def kick_drift_up_simultaneously_unrolled(dphi: np.ndarray, denergy: np.ndarray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> Tuple[np.ndarray, np.ndarray]:
    for i in prange(n_particles):
        dphi[i] -= drift_coef * denergy[i]
        denergy[i] += rfv1 * math.sin(dphi[i] + phi0) \
            + rfv2 * math.sin(h_ratio * (dphi[i] + phi0 - phi12)) - acc_kick
    return dphi, denergy


def kick(machine: object, denergy: np.ndarray,
         dphi: np.ndarray,
         rfv1: np.ndarray,
         rfv2: np.ndarray, npart: int, turn: int,
         up: bool = ...) -> np.ndarray:

    if up:
        return kick_up(dphi, denergy, rfv1[turn], rfv2[turn],
                        machine.phi0[turn], machine.phi12, machine.h_ratio, npart,
                        machine.deltaE0[turn])
    else:
        return kick_down(dphi, denergy, rfv1[turn], rfv2[turn],
                        machine.phi0[turn], machine.phi12, machine.h_ratio, npart,
                        machine.deltaE0[turn])

def kick_and_drift(xp: np.ndarray, yp: np.ndarray,
                   denergy: np.ndarray, dphi: np.ndarray,
                   rfv1: np.ndarray, rfv2: np.ndarray, rec_prof: int,
                   nturns: int, nparts: int,
                   phi0: np.ndarray = None,
                   deltaE0: np.ndarray = None,
                   drift_coef: np.ndarray = None,
                   phi12: float = None,
                   h_ratio: float = None,
                   dturns: int = None,
                   deltaturn: int = None,
                   machine: 'Machine' = None,
                   ftn_out: bool = False, mode: Mode = Mode.CPP) -> Tuple[np.ndarray, np.ndarray]:
    if mode == mode.CUPY:
        from .kick_and_drift_cupy import kick_and_drift_cupy
        return kick_and_drift_cupy(xp, yp, denergy, dphi, rfv1, rfv2, rec_prof, nturns, nparts,
                                phi0, deltaE0, drift_coef, phi12, h_ratio, dturns, deltaturn, machine, ftn_out)

    drift_up_func = drift_up
    drift_down_func = drift_down
    kick_up_func = kick_up
    kick_down_func = kick_down
    kick_drift_up_simultaneously_func = kick_drift_up_simultaneously

    if mode == mode.JIT:
        drift_up_func = njit()(drift_up)
        drift_down_func = njit()(drift_down)
        kick_up_func = njit()(kick_up)
        kick_down_func = njit()(kick_down)
        kick_drift_up_simultaneously_func = njit()(kick_drift_up_simultaneously)
    elif mode == mode.JIT_PARALLEL:
        drift_up_func = njit(parallel=True)(drift_up)
        drift_down_func = njit(parallel=True)(drift_down)
        kick_up_func = njit(parallel=True)(kick_up)
        kick_down_func = njit(parallel=True)(kick_down)
        kick_drift_up_simultaneously_func = njit(parallel=True)(kick_drift_up_simultaneously)
    elif mode == mode.UNROLLED:
        drift_up_func = njit()(drift_up_unrolled)
        drift_down_func = njit()(drift_down_unrolled)
        kick_up_func = njit()(kick_up_unrolled)
        kick_down_func = njit()(kick_down_unrolled)
    elif mode == mode.UNROLLED_PARALLEL:
        drift_up_func = njit(parallel=True)(drift_up_unrolled_parallel)
        drift_down_func = njit(parallel=True)(drift_down_unrolled_parallel)
        kick_up_func = njit(parallel=True)(kick_up_unrolled_parallel)
        kick_down_func = njit(parallel=True)(kick_down_unrolled_parallel)
        kick_drift_up_simultaneously_func = njit(parallel=True)(kick_drift_up_simultaneously_unrolled)
    elif mode == mode.VECTORIZE:
        drift_up_func = vectorize(drift_up_vectorized)
        drift_down_func = vectorize(drift_down_vectorized)
        kick_up_func = vectorize(kick_up_vectorized)
        kick_down_func = vectorize(kick_down_vectorized)
    elif mode == mode.VECTORIZE_PARALLEL:
        drift_up_func = vectorize('float64(float64, float64, float64, int32)', target='parallel')(drift_up_vectorized)
        drift_down_func = vectorize('float64(float64, float64, float64, int32)', target='parallel')(drift_down_vectorized)
        kick_up_func = vectorize('float64(float64, float64, float64, float64, float64, \
                                 float64, float64, int32, float64)', target='parallel')(kick_up_vectorized)
        kick_down_func = vectorize('float64(float64, float64, float64, float64, float64, \
                                 float64, float64, int32, float64)', target='parallel')(kick_down_vectorized)
    elif mode == mode.CPP:
        drift_up_func = drift_up_cpp
        drift_down_func = drift_down_cpp
        kick_up_func = kick_up_cpp
        kick_down_func = kick_down_cpp


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

    progress = 0 # nur für Callback benötigt?
    total = nturns

    together = False

    dphi2 = np.copy(dphi)
    denergy2 = np.copy(denergy)

    while turn < nturns:
        if together:
            turn += 1
            dphi, denergy = kick_drift_up_simultaneously_func(dphi, denergy, drift_coef[turn-1], rfv1[turn], rfv2[turn], phi0[turn],
                                                      phi12_arr[turn], h_ratio, nparts, deltaE0[turn])
        else:
            dphi = drift_up_func(dphi, denergy, drift_coef[turn], nparts)
            dphi2 = drift_up(dphi2, denergy, drift_coef[turn], nparts)

            turn += 1
            denergy = kick_up_func(dphi, denergy, rfv1[turn], rfv2[turn], phi0[turn], phi12_arr[turn],
                h_ratio, nparts, deltaE0[turn])
            denergy2 = kick_up(dphi, denergy2, rfv1[turn], rfv2[turn], phi0[turn], phi12_arr[turn],
                h_ratio, nparts, deltaE0[turn])

            if not np.array_equal(dphi, dphi2) and ftn_out:
                print("Drift up wrong result")

            if not np.array_equal(denergy, denergy2) and ftn_out:
                print("Kick up wrong result")

        if turn % dturns == 0:
            profile += 1
            xp[profile] = np.copy(dphi)
            yp[profile] = np.copy(denergy)

            if ftn_out:
                log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                        0.000% went outside the image width.")
            #callback
        # end while

    profile = rec_prof
    turn = rec_prof * dturns

    if profile > 0:
        # going back to initial coordinates
        dphi = np.copy(xp[rec_prof])
        denergy = np.copy(yp[rec_prof])

        dphi2 = np.copy(xp[rec_prof])
        denergy2 = np.copy(yp[rec_prof])
        # Downwards
        while turn > 0:
            denergy = kick_down_func(dphi, denergy, rfv1[turn], rfv2[turn], phi0[turn],
                    phi12_arr[turn], h_ratio, nparts, deltaE0[turn])

            denergy2 = kick_down(dphi, denergy2, rfv1[turn], rfv2[turn], phi0[turn],
                    phi12_arr[turn], h_ratio, nparts, deltaE0[turn])

            turn -= 1

            dphi = drift_down_func(dphi, denergy, drift_coef[turn], nparts)

            dphi2 = drift_down(dphi2, denergy, drift_coef[turn], nparts)

            if (turn % dturns == 0):
                profile -= 1
                xp[profile] = np.copy(dphi)
                yp[profile] = np.copy(denergy)

                if ftn_out:
                    log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                                0.000% went outside the image width.")
            #callback
        # end while
    return xp, yp

def kick_down(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> np.ndarray:
    denergy -= rfv1 * np.sin(dphi + phi0) \
                      + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick
    return denergy

def kick_down_unrolled(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> np.ndarray:
    for i in range(n_particles):
        denergy[i] -= rfv1 * math.sin(dphi[i] + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi[i] + phi0 - phi12)) - acc_kick
    return denergy

def kick_down_unrolled_parallel(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> np.ndarray:
    for i in prange(n_particles):
        denergy[i] -= rfv1 * math.sin(dphi[i] + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi[i] + phi0 - phi12)) - acc_kick
    return denergy

def kick_down_vectorized(dphi: float, denergy: float, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> float:
    denergy -= rfv1 * math.sin(dphi + phi0) \
                      + rfv2 * math.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick
    return denergy
    denergy -= rfv1 * math.sin(dphi + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick

def kick_down_cpp(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> np.ndarray:
    libtomo.kick_down(dphi, denergy, rfv1, rfv2, phi0, phi12, h_ratio,\
                      n_particles, acc_kick)
    return denergy

def kick_up(dphi: np.ndarray,
            denergy: np.ndarray, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> np.ndarray:
    denergy += rfv1 * np.sin(dphi + phi0) \
                  + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick
    return denergy

def kick_up_unrolled(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> np.ndarray:
    for i in range(n_particles):
        denergy[i] += rfv1 * math.sin(dphi[i] + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi[i] + phi0 - phi12)) - acc_kick
    return denergy

def kick_up_unrolled_parallel(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> np.ndarray:
    for i in prange(n_particles):
        denergy[i] += rfv1 * math.sin(dphi[i] + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi[i] + phi0 - phi12)) - acc_kick
    return denergy

def kick_up_vectorized(dphi: float, denergy: float, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> float:
    denergy += rfv1 * math.sin(dphi + phi0) \
                      + rfv2 * math.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick
    return denergy
    denergy += rfv1 * math.sin(dphi + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick

def kick_up_cpp(dphi: np.ndarray,
            denergy: np.ndarray, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> np.ndarray:
    libtomo.kick_up(dphi, denergy, rfv1, rfv2, phi0, phi12, h_ratio,\
                      n_particles, acc_kick)
    return denergy