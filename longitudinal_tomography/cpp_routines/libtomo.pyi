from typing import Any, Optional

import numpy

def back_project(weights: numpy.ndarray[numpy.float64],
                 flat_points: numpy.ndarray[numpy.int32],
                 flat_profiles: numpy.ndarray[numpy.float64],
                 n_particles: int,
                 n_profiles: int) -> numpy.ndarray[numpy.float64]:
    ...

def drift(denergy: numpy.ndarray[numpy.float64],
          dphi: numpy.ndarray[numpy.float64],
          drift_coef: numpy.ndarray[numpy.float64], npart: int, turn: int,
          up: bool = ...) -> numpy.ndarray[numpy.float64]:
    ...

def drift_down(dphi: numpy.ndarray[numpy.float64],
               denergy: numpy.ndarray[numpy.float64], drift_coef: float,
               n_particles: int) -> None:
    ...

def drift_up(dphi: numpy.ndarray[numpy.float64],
             denergy: numpy.ndarray[numpy.float64], drift_coef: float,
             n_particles: int) -> None:
    ...

def kick(machine: object, denergy: numpy.ndarray[numpy.float64],
         dphi: numpy.ndarray[numpy.float64],
         rfv1: numpy.ndarray[numpy.float64],
         rfv2: numpy.ndarray[numpy.float64], npart: int, turn: int,
         up: bool = ...) -> numpy.ndarray[numpy.float64]:
    ...

def kick_and_drift(*args, **kwargs) -> Any:
    ...

def kick_down(dphi: numpy.ndarray[numpy.float64],
              denergy: numpy.ndarray[numpy.float64], rfv1: float, rfv2: float,
              phi0: float, phi12: float, h_ratio: float, n_particles: int,
              acc_kick: float) -> None:
    ...

def kick_up(dphi: numpy.ndarray[numpy.float64],
            denergy: numpy.ndarray[numpy.float64], rfv1: float, rfv2: float,
            phi0: float, phi12: float, h_ratio: float, n_particles: int,
            acc_kick: float) -> None:
    ...

def make_phase_space(xp: numpy.ndarray[numpy.int32],
                     yp: numpy.ndarray[numpy.int32],
                     weights: numpy.ndarray[numpy.float64],
                     n_bins: int) -> numpy.ndarray[numpy.float64]:
    ...

def project(flat_rec: numpy.ndarray[numpy.float64],
            flat_points: numpy.ndarray[numpy.int32],
            weights: numpy.ndarray[numpy.float64], n_particles: int,
            n_profiles: int, n_bins: int) -> numpy.ndarray[numpy.float64]:
    ...

def reconstruct(xp: numpy.ndarray[numpy.int32],
                waterfall: numpy.ndarray[numpy.float64], n_iter: int,
                n_bins: int, n_particles: int, n_profiles: int,
                verbose: bool = ...,
                callback: Optional[object] = ...) -> tuple:
    ...

def reconstruct_multi(xpRound0: numpy.ndarray[numpy.int32],
                      waterfall: numpy.ndarray[numpy.float64],
                      cutleft: numpy.ndarray[numpy.int32],
                      cutright: numpy.ndarray[numpy.int32],
                      centers: numpy.ndarray[numpy.int32],
                      n_iter: int, n_bins: int, n_particles: int,
                      n_profiles: int, n_centers: int,
                      verbose: bool = ...,
                      callback: Optional[object] = ...) -> tuple:
    ...

def set_num_threads(num_threads: int) -> None:
    ...

def make_phase_space(xp: numpy.ndarray[numpy.int32],
                     yp: numpy.ndarray[numpy.int32],
                     weight: numpy.ndarray[numpy.float64],
                     n_particles: int,
                     n_bins: int) -> numpy.ndarray[numpy.float64]:
    ...