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

# incredibly slow, probably only works with vectorization?
def kick_drift_up_simultaneously_unrolled(dphi: cp.ndarray, denergy: cp.ndarray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> Tuple[cp.ndarray, cp.ndarray]:
    for i in range(n_particles):
        dphi[i] -= drift_coef * denergy[i]
        denergy[i] += (rfv1 * cp.sin(dphi[i] + phi0) \
                  + rfv2 * cp.sin(h_ratio * (dphi[i] + phi0 - phi12)) - acc_kick)
    return dphi, denergy

def kick_drift_down_simultaneously(dphi: cp.ndarray, denergy: cp.ndarray, drift_coef: float, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int, acc_kick: float) -> Tuple[cp.ndarray, cp.ndarray]:
    denergy -= (rfv1 * cp.sin(dphi + phi0) \
                  + rfv2 * cp.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick)
    dphi += drift_coef * denergy
    return dphi, denergy

def kick_and_drift_cupy(xp: np.ndarray, yp: np.ndarray,
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

    xp_gpu = cp.asarray(xp)
    yp_gpu = cp.asarray(yp)
    dphi_gpu = cp.asarray(dphi)
    denergy_gpu = cp.asarray(denergy)

    phi12_arr = np.full(nturns+1, phi12)
    # Preparation end

    profile = rec_prof
    turn = rec_prof * dturns + deltaturn

    if deltaturn < 0:
        profile -= 1

    # Value-based copy to avoid side-effects
    xp_gpu[profile] = cp.copy(dphi_gpu)
    yp_gpu[profile] = cp.copy(denergy_gpu)

    together = True

    while turn < nturns:
        if together:
            turn += 1
            dphi_gpu, denergy_gpu = kick_drift_up_simultaneously(dphi_gpu, denergy_gpu, drift_coef[turn-1], rfv1[turn], rfv2[turn],
                                                              phi0[turn], phi12_arr[turn], h_ratio, nparts, deltaE0[turn])
        else:
            dphi_gpu = drift_up(dphi_gpu, denergy_gpu, drift_coef[turn], nparts)
            turn += 1
            denergy_gpu = kick_up(dphi_gpu, denergy_gpu, rfv1[turn], rfv2[turn], phi0[turn], phi12_arr[turn],
                h_ratio, nparts, deltaE0[turn])

        if turn % dturns == 0:
            profile += 1

            xp_gpu[profile] = cp.copy(dphi_gpu)
            yp_gpu[profile] = cp.copy(denergy_gpu)

        if ftn_out:
            log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                    0.000% went outside the image width.")

    profile = rec_prof
    turn = rec_prof * dturns

    if profile > 0:
        # going back to initial coordinates
        dphi_gpu = cp.copy(xp_gpu[rec_prof])
        denergy_gpu = cp.copy(yp_gpu[rec_prof])

        # Downwards
        while turn > 0:
            if together:
                dphi_gpu, denergy_gpu = kick_drift_down_simultaneously(dphi_gpu, denergy_gpu, drift_coef[turn-1], rfv1[turn], rfv2[turn],
                                                                       phi0[turn], phi12_arr[turn], h_ratio, nparts, deltaE0[turn])
                turn -= 1
            else:
                denergy_gpu = kick_down(dphi_gpu, denergy_gpu, rfv1[turn], rfv2[turn], phi0[turn],
                        phi12_arr[turn], h_ratio, nparts, deltaE0[turn])
                turn -= 1

                dphi_gpu = drift_down(dphi_gpu, denergy_gpu, drift_coef[turn], nparts)

            if (turn % dturns == 0):
                profile -= 1
                xp_gpu[profile] = cp.copy(dphi_gpu)
                yp_gpu[profile] = cp.copy(denergy_gpu)

                if ftn_out:
                    log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                                0.000% went outside the image width.")

    xp = cp.asnumpy(xp_gpu)
    yp = cp.asnumpy(yp_gpu)
    dphi = cp.asnumpy(dphi_gpu)
    denergy = cp.asnumpy(denergy_gpu)

    return xp, yp