"""Module containing the kick-and-drift algorithm with CUDA kernels.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
import cupy as cp
import logging
from typing import Tuple
from ..utils import gpu_dev

log = logging.getLogger(__name__)

if gpu_dev is None:
        from ..utils import GPUDev
        gpu_dev = GPUDev()

kick_down_kernel = gpu_dev.kd_mod.get_function("kick_down")
kick_up_kernel = gpu_dev.kd_mod.get_function("kick_up")
drift_down_kernel = gpu_dev.kd_mod.get_function("drift_down")
drift_up_kernel = gpu_dev.kd_mod.get_function("drift_up")
kick_drift_down_simultaneously_kernel = gpu_dev.kd_mod.get_function("kick_drift_down_simultaneously")
kick_drift_up_simultaneously_kernel = gpu_dev.kd_mod.get_function("kick_drift_up_simultaneously")

block_size = gpu_dev.block_size
grid_size = gpu_dev.grid_size


def drift_down(dphi: cp.ndarray,
               denergy: cp.ndarray, drift_coef: float,
               n_particles: int) -> cp.ndarray:
    drift_down_kernel(args=(dphi, denergy, drift_coef, n_particles),
                      block=block_size, grid=grid_size)
    return dphi

def drift_up(dphi: cp.ndarray,
             denergy: cp.ndarray, drift_coef: float,
             n_particles: int) -> cp.ndarray:
    drift_up_kernel(args=(dphi, denergy, drift_coef, n_particles),
                    block=block_size, grid=grid_size)
    return dphi

def kick_down(dphi: cp.ndarray,
              denergy: cp.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> cp.ndarray:
    kick_down_kernel(args=(dphi, denergy, rfv1, rfv2, phi0, phi12, h_ratio, n_particles, acc_kick),
                     block=block_size, grid=grid_size)
    return denergy

def kick_up(dphi: cp.ndarray,
            denergy: cp.ndarray, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> cp.ndarray:
    kick_up_kernel(args=(dphi, denergy, rfv1, rfv2, phi0, phi12, h_ratio, n_particles, acc_kick),
                   block=block_size, grid=grid_size)
    return denergy

def kick_drift_up_simultaneously(dphi: cp.ndarray, denergy: cp.ndarray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> Tuple[cp.ndarray, cp.ndarray]:
    kick_drift_up_simultaneously_kernel(args=(dphi, denergy, drift_coef, rfv1, rfv2,
                                              phi0, phi12, h_ratio, n_particles, acc_kick),
                                        block=block_size, grid=grid_size)
    return dphi, denergy

def kick_drift_down_simultaneously(dphi: cp.ndarray, denergy: cp.ndarray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> Tuple[cp.ndarray, cp.ndarray]:
    kick_drift_down_simultaneously_kernel(args=(dphi, denergy, drift_coef, rfv1, rfv2,
                                              phi0, phi12, h_ratio, n_particles, acc_kick),
                                        block=block_size, grid=grid_size)
    return dphi, denergy

def kick_and_drift_cuda(xp: cp.ndarray, yp: cp.ndarray,
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

    phi12_arr = np.full(nturns+1, phi12)
    # Preparation end

    profile = rec_prof
    turn = rec_prof * dturns + deltaturn

    if deltaturn < 0:
        profile -= 1

    # Value-based copy to avoid side-effects
    xp[profile] = cp.copy(dphi)
    yp[profile] = cp.copy(denergy)

    together = False

    while turn < nturns:
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