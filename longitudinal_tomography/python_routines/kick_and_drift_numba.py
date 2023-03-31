"""Module containing the kick-and-drift algorithm derived from the cpp functions.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
from typing import Tuple
import logging
from numba import jit

log = logging.getLogger(__name__)

@jit(nopython=True, parallel=True)
def drift(denergy: np.ndarray,
          dphi: np.ndarray,
          drift_coef: np.ndarray, npart: int, turn: int,
          up: bool = ...) -> np.ndarray:
    if up:
        drift_up(dphi, denergy, drift_coef[turn], npart)
    else:
        drift_down(dphi, denergy, drift_coef[turn], npart)
    return dphi

@jit(nopython=True, parallel=True)
def drift_down(dphi: np.ndarray,
               denergy: np.ndarray, drift_coef: float,
               n_particles: int) -> None:
    dphi += drift_coef * denergy

@jit(nopython=True, parallel=True)
def drift_up(dphi: np.ndarray,
             denergy: np.ndarray, drift_coef: float,
             n_particles: int) -> None:
    dphi -= drift_coef * denergy

@jit(nopython=True, parallel=True)
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
                   ftn_out: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    # PREPARATION: see libtomo.kick_and_drift_machine/scalar/array
    phi12_arr = np.full(nturns+1, phi12) # nturns+1 so it has the same dimension as the others
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
        drift_up(dphi, denergy, drift_coef[turn], nparts)

        turn += 1

        kick_up(dphi, denergy, rfv1[turn], rfv2[turn], phi0[turn], phi12_arr[turn],
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
            print("kickdown")
            kick_down(dphi, denergy, rfv1[turn], rfv2[turn], phi0[turn],
                    phi12_arr[turn], h_ratio, nparts, deltaE0[turn])

            turn -= 1

            drift_down(dphi, denergy, drift_coef[turn], nparts)

        if (turn % dturns == 0):
            profile -= 1

        xp[profile] = dphi
        yp[profile] = denergy

        if ftn_out:
            log.info(f"Tracking from time slice {rec_prof + 1} to {profile + 1},\
                        0.000% went outside the image width.")
        #callback
    # end while

@jit(nopython=True, parallel=True)
def kick_down(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    denergy -= rfv1 * np.sin(dphi + phi0) \
                  + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick

@jit(nopython=True, parallel=True)
def kick_up(dphi: np.ndarray,
            denergy: np.ndarray, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> None:
    denergy += rfv1 * np.sin(dphi + phi0) \
                  + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick