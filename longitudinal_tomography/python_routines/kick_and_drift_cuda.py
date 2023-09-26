"""Module containing the kick-and-drift algorithm with CUDA kernels.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
import cupy as cp
import logging
from typing import Tuple
from ..utils import gpu_dev

import os

log = logging.getLogger(__name__)

### Testing purposes - remove later

if gpu_dev is None:
        from ..utils import GPUDev
        gpu_dev = GPUDev()

if os.getenv('SINGLE_PREC') is not None:
    single_precision = True if os.getenv('SINGLE_PREC') == 'True' else False
else:
    single_precision = False

if single_precision:
    kick_drift_up_turns = gpu_dev.kd_mod.get_function("kick_drift_up_turns_float")
    kick_drift_down_turns = gpu_dev.kd_mod.get_function("kick_drift_down_turns_float")
else:
    kick_drift_up_turns = gpu_dev.kd_mod.get_function("kick_drift_up_turns_double")
    kick_drift_down_turns = gpu_dev.kd_mod.get_function("kick_drift_down_turns_double")

#kick_drift_up_turns = gpu_dev.kd_mod.get_function("kick_drift_up_turns")
#kick_drift_down_turns = gpu_dev.kd_mod.get_function("kick_drift_down_turns")

block_size = gpu_dev.block_size
grid_size = gpu_dev.grid_size

def kick_drift_up_whole(dphi: cp.ndarray, denergy: cp.ndarray, xp: cp.ndarray, yp: cp.ndarray, drift_coef: cp.ndarray,
                        rfv1: cp.ndarray, rfv2: cp.ndarray, phi0: cp.ndarray, phi12: cp.ndarray, h_ratio: float,
                        n_particles: int, acc_kick: cp.ndarray, turn: int, nturns: int, dturns: int, profile: int) -> None:
    kick_drift_up_turns(args=(dphi, denergy, xp, yp, drift_coef, rfv1, rfv2,
                            phi0, phi12, h_ratio, n_particles, acc_kick,
                            turn, nturns, dturns, profile),
                        block=block_size, grid=grid_size)

def kick_drift_down_whole(dphi: cp.ndarray, denergy: cp.ndarray, xp: cp.ndarray, yp: cp.ndarray, drift_coef: cp.ndarray,
                        rfv1: cp.ndarray, rfv2: cp.ndarray, phi0: cp.ndarray, phi12: cp.ndarray, h_ratio: float,
                        n_particles: int, acc_kick: cp.ndarray, turn: int, dturns: int, profile: int) -> None:
    kick_drift_down_turns(args=(dphi, denergy, xp, yp, drift_coef, rfv1, rfv2,
                            phi0, phi12, h_ratio, n_particles, acc_kick,
                            turn, dturns, profile),
                        block=block_size, grid=grid_size)

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
                   deltaturn: int) -> Tuple[cp.ndarray, cp.ndarray]:
    global grid_size
    grid_size = (int(nparts / block_size[0]) + 1, 1, 1)

    if single_precision:
        xp = xp.astype(cp.float32)
        yp = yp.astype(cp.float32)
        denergy = denergy.astype(cp.float32)
        dphi = dphi.astype(cp.float32)
        drift_coef = drift_coef.astype(np.float32)
        phi0 = phi0.astype(np.float32)
        deltaE0 = deltaE0.astype(np.float32)
        rfv1 = rfv1.astype(np.float32)
        rfv2 = rfv2.astype(np.float32)

    phi12_arr = np.full(nturns+1, phi12)
    # Preparation end

    profile = rec_prof
    turn = rec_prof * dturns + deltaturn

    if deltaturn < 0:
        profile -= 1

    # Value-based copy to avoid side-effects
    xp[profile] = cp.copy(dphi)
    yp[profile] = cp.copy(denergy)

    kick_drift_up_whole(dphi, denergy, xp, yp, cp.asarray(drift_coef), cp.asarray(rfv1), cp.asarray(rfv2),
                        cp.asarray(phi0), cp.asarray(phi12_arr), h_ratio, nparts, cp.asarray(deltaE0),
                        turn, nturns, dturns, profile)

    profile = rec_prof
    turn = rec_prof * dturns


    if profile > 0:
        # going back to initial coordinates
        dphi = cp.copy(xp[rec_prof])
        denergy = cp.copy(yp[rec_prof])

        kick_drift_down_whole(dphi, denergy, xp, yp, cp.asarray(drift_coef), cp.asarray(rfv1), cp.asarray(rfv2),
                        cp.asarray(phi0), cp.asarray(phi12_arr), h_ratio, nparts, cp.asarray(deltaE0),
                        turn, dturns, profile)
    return xp, yp