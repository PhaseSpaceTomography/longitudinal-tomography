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

def normalize(flat_rec: numpy.ndarray[numpy.float64],
              n_profiles: int, n_bins: int) -> numpy.ndarray[numpy.float64]:
    ...

def clip(array: numpy.ndarray[numpy.float64],
         length: int, clip_val: float) -> numpy.ndarray[numpy.float64]:
    ...

def find_difference_profile(diff_profiles: numpy.ndarray[numpy.float64],
                            flat_rec: numpy.ndarray[numpy.float64],
                            flat_profiles: numpy.ndarray[numpy.float64],
                            all_bins: int) -> numpy.ndarray[numpy.float64]:
    ...

def discrepancy(diff_profiles: numpy.ndarray[numpy.float64],
                n_profiles: int, n_bins: int) -> float:
    ...

def compensate_particle_amount(diff_profiles: numpy.ndarray[numpy.float64],
                               rparts: numpy.ndarray[numpy.float64],
                               n_profiles: int, n_bins: int) -> numpy.ndarray[numpy.float64]:
    ...

def max_2d(array: numpy.ndarray[numpy.float64], x_axis: int, y_axis: int) -> float:
    ...

def max_1d(array: numpy.ndarray[numpy.float64], length: int) -> float:
    ...

def count_particles_in_bin(rparts: numpy.ndarray[numpy.float64], xp: numpy.ndarray[numpy.int32],
                           n_profiles: int, n_particles: int, n_bins: int) -> numpy.ndarray[numpy.float64]:
    ...

def reciprocal_particles(rparts: numpy.ndarray[numpy.float64], xp: numpy.ndarray[numpy.int32],
                          n_bins: int, n_profiles: int, n_particles: int) -> numpy.ndarray[numpy.float64]:
    ...

def create_flat_points(xp: numpy.ndarray[numpy.int32], flat_points: numpy.ndarray[numpy.int32],
                       n_particles: int, n_profiles: int, n_bins: int) -> numpy.ndarray[numpy.int32]:
    ...

def set_num_threads(num_threads: int) -> None:
    ...
