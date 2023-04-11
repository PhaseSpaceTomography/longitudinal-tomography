"""Module containing the kick-and-drift algorithm derived from the cpp functions.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
from typing import Tuple
import logging
from enum import Enum
from numba import njit, vectorize, prange
import math
from ..cpp_routines import libtomo

log = logging.getLogger(__name__)

class Mode(Enum):
    PURE = 1 # numpy
    JIT = 2
    JIT_PARALLEL = 3
    UNROLLED = 4
    UNROLLED_PARALLEL = 5
    VECTORIZE = 6
    VECTORIZE_PARALLEL = 7
    CPP = 8

def drift(denergy: np.ndarray,
          dphi: np.ndarray,
          drift_coef: np.ndarray, npart: int, turn: int,
          up: bool = ...) -> np.ndarray:
    if up:
        drift_up(dphi, denergy, drift_coef[turn], npart)
    else:
        drift_down(dphi, denergy, drift_coef[turn], npart)
    return dphi

def drift_down(dphi: np.ndarray,
               denergy: np.ndarray, drift_coef: float,
               n_particles: int) -> None:
    dphi += drift_coef * denergy

def drift_down_unrolled(dphi: np.ndarray,
                        denergy: np.ndarray, drift_coef: float,
                        n_particles: int) -> None:
    for i in range(n_particles):
        dphi[i] += drift_coef * denergy[i]

def drift_down_unrolled_parallel(dphi: np.ndarray,
                                 denergy: np.ndarray, drift_coef: float,
                                 n_particles: int) -> None:
    for i in prange(n_particles):
        dphi[i] += drift_coef * denergy[i]

def drift_down_vectorized(dphi: float, denergy: float, drift_coef: float) -> None:
    dphi += drift_coef * denergy

def drift_down_cpp(dphi: np.ndarray,
                   denergy: np.ndarray, drift_coef: float,
                   n_particles: int) -> None:
    libtomo.drift_down(dphi, denergy, drift_coef, n_particles)


def drift_up(dphi: np.ndarray,
             denergy: np.ndarray, drift_coef: float,
             n_particles: int) -> None:
    dphi -= drift_coef * denergy

def drift_up_unrolled(dphi: np.ndarray,
                      denergy: np.ndarray, drift_coef: float,
                      n_particles: int) -> None:
    for i in range(n_particles):
        dphi[i] -= drift_coef * denergy[i]

def drift_up_unrolled_parallel(dphi: np.ndarray,
                      denergy: np.ndarray, drift_coef: float,
                      n_particles: int) -> None:
    for i in prange(n_particles):
        dphi[i] -= drift_coef * denergy[i]

def drift_up_vectorized(dphi: float, denergy: float, drift_coef: float) -> None:
    dphi -= drift_coef * denergy

def drift_up_cpp(dphi: np.ndarray,
                   denergy: np.ndarray, drift_coef: float,
                   n_particles: int) -> None:
    libtomo.drift_up(dphi, denergy, drift_coef, n_particles)

def kick(machine: object, denergy: np.ndarray,
         dphi: np.ndarray,
         rfv1: np.ndarray,
         rfv2: np.ndarray, npart: int, turn: int,
         up: bool = ...) -> np.ndarray:

    if up:
        kick_up(dphi, denergy, rfv1[turn], rfv2[turn],
                        machine.phi0[turn], machine.phi12, machine.h_ratio, npart,
                        machine.deltaE0[turn])
    else:
        kick_down(dphi, denergy, rfv1[turn], rfv2[turn],
                        machine.phi0[turn], machine.phi12, machine.h_ratio, npart,
                        machine.deltaE0[turn])

    return denergy

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
                   ftn_out: bool = False, mode: Mode = Mode.PURE) -> Tuple[np.ndarray, np.ndarray]:

    drift_up_func = drift_up
    drift_down_func = drift_down
    kick_up_func = kick_up
    kick_down_func = kick_down

    if mode == mode.JIT:
        drift_up_func = njit()(drift_up)
        drift_down_func = njit()(drift_down)
        kick_up_func = njit()(kick_up)
        kick_down_func = njit()(kick_down_func)
    elif mode == mode.JIT_PARALLEL:
        drift_up_func = njit(parallel=True)(drift_up)
        drift_down_func = njit(parallel=True)(drift_down)
        kick_up_func = njit(parallel=True)(kick_up)
        kick_down_func = njit(parallel=True)(kick_down)
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
    elif mode == mode.VECTORIZE:
        drift_up_func = vectorize(drift_up_vectorized)
        drift_down_func = vectorize(drift_down_vectorized)
        kick_up_func = vectorize(kick_up_vectorized)
        kick_down_func = vectorize(kick_down_vectorized)
    elif mode == mode.VECTORIZE_PARALLEL:
        drift_up_func = vectorize('void(float64, float64, float64)', target='parallel')(drift_up_vectorized)
        drift_down_func = vectorize('void(float64, float64, float64)', target='parallel')(drift_down_vectorized)
        kick_up_func = vectorize('void(float64, float64, float64, float64, float64, \
                                 float64, float64, int32, float64)', target='parallel')(kick_up_vectorized)
        kick_down_func = vectorize('void(float64, float64, float64, float64, float64, \
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
    turn = rec_prof * dturns + dturns

    if dturns < 0:
        profile -= 1

    xp[profile] = dphi
    yp[profile] = denergy

    progress = 0 # nur für Callback benötigt?
    total = nturns

    while turn < nturns:
        drift_up_func(dphi, denergy, drift_coef[turn], nparts)

        turn += 1

        kick_up_func(dphi, denergy, rfv1[turn], rfv2[turn], phi0[turn], phi12_arr[turn],
                h_ratio, nparts, deltaE0[turn])

        if turn % dturns == 0:
            profile += 1

        xp[profile] = dphi
        yp[profile] = denergy

        if ftn_out:
            log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                    0.000% went outside the image width.")
            #callback
        # end while

    profile = rec_prof
    turn = rec_prof * dturns

    if profile > 0:
        # going back to initial coordinates
        dphi = xp[rec_prof]
        denergy = yp[rec_prof]

        # Downwards
        while turn > 0:
            kick_down_func(dphi, denergy, rfv1[turn], rfv2[turn], phi0[turn],
                    phi12_arr[turn], h_ratio, nparts, deltaE0[turn])
            turn -= 1

            drift_down_func(dphi, denergy, drift_coef[turn], nparts)

        if (turn % dturns == 0):
            profile -= 1

        xp[profile] = dphi
        yp[profile] = denergy

        if ftn_out:
            log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                        0.000% went outside the image width.")
        #callback
    # end while

def kick_down(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    denergy -= rfv1 * np.sin(dphi + phi0) \
                      + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick

def kick_down_unrolled(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    for i in range(n_particles):
        denergy[i] -= rfv1 * math.sin(dphi[i] + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi[i] + phi0 - phi12)) - acc_kick

def kick_down_unrolled_parallel(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    for i in prange(n_particles):
        denergy[i] -= rfv1 * math.sin(dphi[i] + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi[i] + phi0 - phi12)) - acc_kick

def kick_down_vectorized(dphi: float, denergy: float, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    denergy -= rfv1 * math.sin(dphi + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick

def kick_down_cpp(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    libtomo.kick_down(dphi, denergy, rfv1, rfv2, phi0, phi12, h_ratio,\
                      n_particles, acc_kick)

def kick_up(dphi: np.ndarray,
            denergy: np.ndarray, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> None:
    denergy += rfv1 * np.sin(dphi + phi0) \
                  + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick

def kick_up_unrolled(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    for i in range(n_particles):
        denergy[i] += rfv1 * math.sin(dphi[i] + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi[i] + phi0 - phi12)) - acc_kick

def kick_up_unrolled_parallel(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    for i in prange(n_particles):
        denergy[i] += rfv1 * math.sin(dphi[i] + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi[i] + phi0 - phi12)) - acc_kick

def kick_up_vectorized(dphi: float, denergy: float, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    denergy += rfv1 * math.sin(dphi + phi0) \
        + rfv2 * math.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick

def kick_up_cpp(dphi: np.ndarray,
            denergy: np.ndarray, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> None:
    libtomo.kick_up(dphi, denergy, rfv1, rfv2, phi0, phi12, h_ratio,\
                      n_particles, acc_kick)