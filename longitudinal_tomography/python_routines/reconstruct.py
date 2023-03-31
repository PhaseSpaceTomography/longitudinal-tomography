"""Module containing the reconstruction algorithm derived from the cpp functions.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np

def back_project(weights: np.ndarray[np.float64],
                 flat_points: np.ndarray[np.int32],
                 flat_profiles: np.ndarray[np.float64],
                 n_particles: int,
                 n_profiles: int) -> np.ndarray[np.float64]:
    for i in range(n_particles):
        for j in range(n_profiles):
            weights[i] += flat_profiles[flat_points[i * n_profiles + j]]
    return weights

def project(flat_rec: np.ndarray[np.float64],
            flat_points: np.ndarray[np.int32],
            weights: np.ndarray[np.float64], n_particles: int,
            n_profiles: int, n_bins: int) -> np.ndarray[np.float64]:
    for i in range(n_particles):
        for j in range(n_profiles):
            flat_rec[flat_points[i * n_profiles + j]] += weights[i]
    return flat_rec

def normalize(flat_rec: np.ndarray[np.float64],
              n_profiles: int, n_bins: int) -> np.ndarray[np.float64]:
    pass

def clip(array: np.ndarray[np.float64],
         length: int, clip_val: float) -> np.ndarray[np.float64]:
    pass

def find_difference_profile(flat_rec: np.ndarray[np.float64],
                            flat_profiles: np.ndarray[np.float64],
                            all_bins: int) -> np.ndarray[np.float64]:
    pass

def discrepancy(diff_prof: np.ndarray[np.float64],
                n_profiles: int, n_bins = int) -> float:
    pass

def compensate_particle_amount(diff_prof: np.ndarray[np.float64],
                               rparts: np.ndarray[np.float64],
                               n_profiles: int, n_bins: int) -> np.ndarray[np.float64]:
    pass

def max_2d(array: np.ndarray[np.float64],
           x_axis: int, y_axis: int) -> float:
    pass

def max_1d(array: np.ndarray[np.float64],
           length: int) -> float:
    pass

def count_particles_in_bin(xp: np.ndarray[np.int32],
                           n_profiles: int, n_particles: int,
                           n_bins: int) -> np.ndarray[np.float64]:
    pass

def reciprocal_particles(xp: np.ndarray[np.int32],
                         n_bins: int, n_profiles: int,
                         n_particles: int) -> np.ndarray[np.float64]:
    pass

def create_flat_points(xp: np.ndarray[np.int32],
                       n_particles: int, n_profiles: int,
                       n_bins: int) -> np.ndarray[np.int32]:
    pass

# TODO

def reconstruct(xp: np.ndarray[np.int32],
                waterfall: np.ndarray[np.float64], n_iter: int,
                n_bins: int, n_particles: int, n_profiles: int,
                verbose: bool = ...,
                callback: Optional[object] = ...) -> tuple:
    ...