"""Module containing the kick-and-drift algorithm derived from the cpp functions.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
from typing import Tuple
import logging
from ..cpp_routines import libtomo

log = logging.getLogger(__name__)

def drift(denergy: np.ndarray,
          dphi: np.ndarray,
          drift_coef: np.ndarray, npart: int, turn: int,
          up: bool = ...) -> np.ndarray:
    libtomo.drift(denergy, dphi, drift_coef, npart, turn, up)
    # if up:
    #     drift_up(dphi, denergy, drift_coef[turn], npart)
    # else:
    #     drift_down(dphi, denergy, drift_coef[turn], npart)
    return dphi

def drift_down(dphi: np.ndarray,
               denergy: np.ndarray, drift_coef: float,
               n_particles: int) -> None:
    libtomo.drift_down(dphi, denergy, drift_coef, n_particles)

    # dphi += drift_coef * denergy

def drift_up(dphi: np.ndarray,
             denergy: np.ndarray, drift_coef: float,
             n_particles: int) -> None:
    libtomo.drift_up(dphi, denergy, drift_coef, n_particles)

    # dphi -= drift_coef * denergy

def kick(machine: object, denergy: np.ndarray,
         dphi: np.ndarray,
         rfv1: np.ndarray,
         rfv2: np.ndarray, npart: int, turn: int,
         up: bool = ...) -> np.ndarray:

    libtomo.kick(machine, denergy, dphi, rfv1, rfv2, npart, turn, up)

    # if up:
    #     kick_up(dphi, denergy, rfv1[turn], rfv2[turn],
    #                     machine.phi0[turn], machine.phi12, machine.h_ratio, npart,
    #                     machine.deltaE0[turn])
    # else:
    #     kick_down(dphi, denergy, rfv1[turn], rfv2[turn],
    #                     machine.phi0[turn], machine.phi12, machine.h_ratio, npart,
    #                     machine.deltaE0[turn])

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

def kick_down(dphi: np.ndarray,
              denergy: np.ndarray, rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    libtomo.kick_down(dphi, denergy, rfv1, rfv2, phi0, phi12, h_ratio,\
                      n_particles, acc_kick)
    # denergy -= rfv1 * np.sin(dphi + phi0) \
    #                   + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick

def kick_up(dphi: np.ndarray,
            denergy: np.ndarray, rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> None:
    libtomo.kick_up(dphi, denergy, rfv1, rfv2, phi0, phi12, h_ratio,\
                      n_particles, acc_kick)
    # denergy += rfv1 * np.sin(dphi + phi0) \
    #               + rfv2 * np.sin(h_ratio * (dphi + phi0 - phi12)) - acc_kick