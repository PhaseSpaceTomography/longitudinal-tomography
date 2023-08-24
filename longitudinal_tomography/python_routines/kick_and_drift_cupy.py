"""Module containing the kick-and-drift algorithm with CuPy.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
import cupy as cp
from cupyx import jit
import logging
from typing import Tuple
from ..utils import gpu_dev
from pyprof import timing

log = logging.getLogger(__name__)

if gpu_dev is None:
        from ..utils import GPUDev
        gpu_dev = GPUDev()

block_size = gpu_dev.block_size
grid_size = gpu_dev.grid_size


def drift_down(dphi: cp.ndarray,
               denergy: cp.ndarray, drift_coef: float,
               n_particles: int) -> cp.ndarray:
    dphi += drift_coef * denergy
    return dphi

@timing.timeit(key='drift_up')
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

@timing.timeit(key='kick_up')
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

# Experimental
@timing.timeit(key="tracking::kick_drift_up_whole_cupyjit")
@jit.rawkernel()
def kick_drift_up_whole(dphi: cp.ndarray, denergy: cp.ndarray, xp: cp.ndarray, yp: cp.ndarray, drift_coef: cp.ndarray,
                        rfv1: cp.ndarray, rfv2: cp.ndarray, phi0: cp.ndarray, phi12: cp.ndarray, h_ratio: float,
                        n_particles: int, acc_kick: cp.ndarray, turn: int, nturns: int, dturns: int, profile: int) -> Tuple[cp.ndarray, cp.ndarray]:
    tid = jit.threadIdx.x + jit.blockDim.x * jit.blockIdx.x
    current_dphi = 0.0
    current_denergy = 0.0

    if tid < n_particles:
        current_dphi = dphi[tid]
        current_denergy = denergy[tid]

        while turn < nturns:
            current_dphi -= drift_coef[turn] * current_denergy
            turn += 1
            current_denergy += rfv1[turn] * cp.sin(current_dphi + phi0[turn]) \
                        + rfv2[turn] * cp.sin(h_ratio * (current_dphi + phi0[turn] - phi12[turn])) - acc_kick[turn]

            if turn % dturns == 0:
                profile += 1
                xp[n_particles * profile + tid] = current_dphi
                yp[n_particles * profile + tid] = current_denergy

# Experimental
@timing.timeit(key="tracking::kick_drift_down_whole_cupyjit")
@jit.rawkernel()
def kick_drift_down_whole(dphi: cp.ndarray, denergy: cp.ndarray, xp: cp.ndarray, yp: cp.ndarray, drift_coef: cp.ndarray,
                        rfv1: cp.ndarray, rfv2: cp.ndarray, phi0: cp.ndarray, phi12: cp.ndarray, h_ratio: float,
                        n_particles: int, acc_kick: cp.ndarray, turn: int, nturns: int, dturns: int, profile: int) -> Tuple[cp.ndarray, cp.ndarray]:
    tid = jit.threadIdx.x + jit.blockDim.x * jit.blockIdx.x
    current_dphi = 0.0
    current_denergy = 0.0

    if tid < n_particles:
        current_dphi = dphi[tid]
        current_denergy = denergy[tid]

        while turn > 0:
            current_denergy -= rfv1[turn] * cp.sin(current_dphi + phi0[turn]) \
                        + rfv2[turn] * cp.sin(h_ratio * (current_dphi + phi0[turn] - phi12[turn])) - acc_kick[turn]
            turn -= 1
            current_dphi += drift_coef[turn] * current_denergy

            if turn % dturns == 0:
                profile -= 1
                xp[n_particles * profile + tid] = current_dphi
                yp[n_particles * profile + tid] = current_denergy

def kick_and_drift_cupy(xp: cp.ndarray, yp: cp.ndarray,
                   denergy: cp.ndarray, dphi: cp.ndarray,
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
                   ftn_out: bool = False) -> Tuple[cp.ndarray, cp.ndarray]:
    global grid_size
    grid_size = (int(nparts / block_size[0]) + 1, 1, 1)

    phi12_arr = np.full(nturns+1, phi12)
    # Preparation end

    profile = rec_prof
    turn = rec_prof * dturns + deltaturn

    if deltaturn < 0:
        profile -= 1

    # Value-based copy to avoid side-effects
    xp[profile] = cp.copy(dphi)
    yp[profile] = cp.copy(denergy)

    use_jit = False

    if use_jit:
        nprofs = len(xp)
        xp = xp.reshape(-1)
        yp = yp.reshape(-1)

        kick_drift_up_whole(grid_size, block_size,(dphi, denergy, xp, yp, cp.asarray(drift_coef), cp.asarray(rfv1), cp.asarray(rfv2),
                            cp.asarray(phi0), cp.asarray(phi12_arr), h_ratio, nparts, cp.asarray(deltaE0),
                            turn, nturns, dturns, profile))

        profile = rec_prof
        turn = rec_prof * dturns

        if profile > 0:
            # Downwards
            kick_drift_down_whole(grid_size, block_size, (dphi, denergy, xp, yp, cp.asarray(drift_coef), cp.asarray(rfv1), cp.asarray(rfv2),
                        cp.asarray(phi0), cp.asarray(phi12_arr), h_ratio, nparts, cp.asarray(deltaE0),
                        turn, nturns, dturns, profile))

            xp = xp.reshape((nprofs, nparts))
            yp = yp.reshape((nprofs, nparts))

    else:
        while turn < nturns:
            together = False

            if together:
                turn += 1
                dphi, denergy = kick_drift_up_simultaneously(dphi, denergy, drift_coef[turn-1], rfv1[turn], rfv2[turn],
                                                                phi0[turn], phi12_arr[turn], h_ratio, nparts, deltaE0[turn])
            else:
                dphi = drift_up(dphi, denergy, drift_coef[turn], nparts)
                turn += 1
                denergy = kick_up(dphi, denergy, rfv1[turn], rfv2[turn], phi0[turn], phi12_arr[turn],
                    h_ratio, nparts, deltaE0[turn])

            if turn % dturns == 0:
                profile += 1

                xp[profile] = cp.copy(dphi)
                yp[profile] = cp.copy(denergy)

            if ftn_out:
                log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                        0.000% went outside the image width.")

        profile = rec_prof
        turn = rec_prof * dturns

        if profile > 0:
            # going back to initial coordinates
            dphi = cp.copy(xp[rec_prof])
            denergy = cp.copy(yp[rec_prof])

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
                    xp[profile] = cp.copy(dphi)
                    yp[profile] = cp.copy(denergy)

                    if ftn_out:
                        log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                                    0.000% went outside the image width.")
    return xp, yp