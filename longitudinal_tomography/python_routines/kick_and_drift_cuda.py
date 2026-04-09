"""Module containing the kick-and-drift algorithm with CUDA kernels.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
import cupy as cp
import logging
import os
from typing import Tuple
from ..utils.tomo_config import GPUDev

log = logging.getLogger(__name__)

gpu_dev = GPUDev.get_gpu_dev()
block_size = gpu_dev.block_size
grid_size = gpu_dev.grid_size

def refresh_kernels():
    global kick_drift_up_turns, kick_drift_down_turns
    global kick_drift_up_turns_int, kick_drift_down_turns_int
    global generate_sin_lut
    gpu_dev = GPUDev.get_gpu_dev()
    kick_drift_up_turns = gpu_dev.kd_mod.get_function("kick_drift_up_turns")
    kick_drift_down_turns = gpu_dev.kd_mod.get_function("kick_drift_down_turns")
    kick_drift_up_turns_int = gpu_dev.kd_mod.get_function("kick_drift_up_turns_int")
    kick_drift_down_turns_int = gpu_dev.kd_mod.get_function("kick_drift_down_turns_int")
    generate_sin_lut = gpu_dev.kd_mod.get_function("generate_sin_lut")

def kick_drift_up_whole(dphi: cp.ndarray, denergy: cp.ndarray, xp: cp.ndarray, yp: cp.ndarray, drift_coef: cp.ndarray,
                        rfv1: cp.ndarray, rfv2: cp.ndarray, phi0: cp.ndarray, phi12: cp.ndarray, h_ratio: float,
                        n_particles: int, acc_kick: cp.ndarray, turn: int, nturns: int, dturns: int, profile: int) -> None:
    kick_drift_up_turns(args=(dphi, denergy, xp, yp, drift_coef, rfv1, rfv2,
                            phi0, phi12, h_ratio, n_particles, acc_kick,
                            turn, nturns, dturns, profile),
                        block=block_size, grid=(int(n_particles / block_size[0] + 1), 1, 1))

def kick_drift_down_whole(dphi: cp.ndarray, denergy: cp.ndarray, xp: cp.ndarray, yp: cp.ndarray, drift_coef: cp.ndarray,
                        rfv1: cp.ndarray, rfv2: cp.ndarray, phi0: cp.ndarray, phi12: cp.ndarray, h_ratio: float,
                        n_particles: int, acc_kick: cp.ndarray, turn: int, dturns: int, profile: int) -> None:
    kick_drift_down_turns(args=(dphi, denergy, xp, yp, drift_coef, rfv1, rfv2,
                            phi0, phi12, h_ratio, n_particles, acc_kick,
                            turn, dturns, profile),
                        block=block_size, grid=(int(n_particles / block_size[0] + 1), 1, 1))

def kick_drift_up_whole_int(dphi: cp.ndarray, denergy: cp.ndarray, xp: cp.ndarray, yp: cp.ndarray, drift_coef: cp.ndarray,
                        rfv1: cp.ndarray, rfv2: cp.ndarray, phi0: cp.ndarray, phi12: cp.ndarray, h_ratio: float,
                        n_particles: int, acc_kick: cp.ndarray, turn: int, nturns: int, dturns: int, profile: int,
                        S: int, G: int, x0: float, x1: float, lut: cp.ndarray) -> None:
    try:
        kick_drift_up_turns_int(args=(dphi, denergy, xp, yp, drift_coef, rfv1, rfv2,
                            phi0, phi12, h_ratio, n_particles, acc_kick,
                            turn, nturns, dturns, profile, S, G, x0, x1, lut),
                        block=block_size, grid=grid_size
                        )
        cp.cuda.Stream.null.synchronize()
    except Exception as e:
        print(e)
        os._exit(1)

def kick_drift_down_whole_int(dphi: cp.ndarray, denergy: cp.ndarray, xp: cp.ndarray, yp: cp.ndarray, drift_coef: cp.ndarray,
                        rfv1: cp.ndarray, rfv2: cp.ndarray, phi0: cp.ndarray, phi12: cp.ndarray, h_ratio: float,
                        n_particles: int, acc_kick: cp.ndarray, turn: int, dturns: int, profile: int,
                        S: int, G: int, x0: float, x1: float, lut: cp.ndarray) -> None:
    try:
        kick_drift_down_turns_int(args=(dphi, denergy, xp, yp, drift_coef, rfv1, rfv2,
                            phi0, phi12, h_ratio, n_particles, acc_kick,
                            turn, dturns, profile, S, G, x0, x1, lut),
                        block=block_size, grid=grid_size
                        )
        cp.cuda.Stream.null.synchronize()
    except Exception as e:
        print(e)
        os._exit(1)

def generate_sin_lut_wrapper(lut: cp.ndarray, x0: float, x1: float, G: int, S: int) -> None:
    generate_sin_lut(args=(lut, x0, x1, G, S),
                        block=block_size, grid=(int(lut.shape[0] / block_size[0] + 1), 1, 1))

def kick_and_drift_cuda(xp: cp.ndarray, yp: cp.ndarray,
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
                   ftn_out: bool,
                   callback
                   ) -> Tuple[cp.ndarray, cp.ndarray]:
    phi12_arr = cp.full(nturns+1, phi12)
    # Preparation end

    profile = rec_prof
    turn = rec_prof * dturns + deltaturn

    if deltaturn < 0:
        profile -= 1

    # Value-based copy to avoid side-effects
    xp[profile] = cp.copy(dphi)
    yp[profile] = cp.copy(denergy)

    kick_drift_up_whole(dphi, denergy, xp, yp, drift_coef, rfv1, rfv2,
                        phi0, phi12_arr, h_ratio, nparts, deltaE0,
                        turn, nturns, dturns, profile)

    profile = rec_prof
    turn = rec_prof * dturns


    if profile > 0:
        # going back to initial coordinates
        dphi = cp.copy(xp[rec_prof])
        denergy = cp.copy(yp[rec_prof])

        kick_drift_down_whole(dphi, denergy, xp, yp, drift_coef, rfv1, rfv2,
                        phi0, phi12_arr, h_ratio, nparts, deltaE0,
                        turn, dturns, profile)

def kick_and_drift_cuda_int(xp: cp.ndarray, yp: cp.ndarray,
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
                   ftn_out: bool,
                   S: int,
                   G: int,
                   x0: float,
                   x1: float,
                   callback
                   ) -> Tuple[cp.ndarray, cp.ndarray]:
    phi12_arr = cp.full(nturns+1, phi12)
    # Preparation end

    # Calulating look-up table
    dx_int = int(round((x1 - x0) / (G - 1) * S))
    if dx_int <= 0:
        raise ValueError("Error in generating the look-up table, `dx_int` <= 0.")

    lut = cp.zeros(G, dtype=xp.dtype)
    generate_sin_lut_wrapper(lut, x0, x1, G, S)

    profile = rec_prof
    turn = rec_prof * dturns + deltaturn

    if deltaturn < 0:
        profile -= 1

    # Value-based copy to avoid side-effects
    xp[profile] = cp.copy(dphi)
    yp[profile] = cp.copy(denergy)

    kick_drift_up_whole_int(dphi, denergy, xp, yp, drift_coef, rfv1, rfv2,
                        phi0, phi12_arr, h_ratio, nparts, deltaE0,
                        turn, nturns, dturns, profile, S, G, x0, x1, lut)

    profile = rec_prof
    turn = rec_prof * dturns


    if profile > 0:
        # going back to initial coordinates
        dphi = cp.copy(xp[rec_prof])
        denergy = cp.copy(yp[rec_prof])

        kick_drift_down_whole_int(dphi, denergy, xp, yp, drift_coef, rfv1, rfv2,
                        phi0, phi12_arr, h_ratio, nparts, deltaE0,
                        turn, dturns, profile, S, G, x0, x1, lut)

refresh_kernels()
