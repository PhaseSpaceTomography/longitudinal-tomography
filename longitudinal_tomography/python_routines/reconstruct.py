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

def project(flat_rec: np.ndarray[np.float64],
            flat_points: np.ndarray[np.int32],
            weights: np.ndarray[np.float64], n_particles: int,
            n_profiles: int, n_bins: int) -> np.ndarray[np.float64]:
    for i in range(n_particles):
        for j in range(n_profiles):
            flat_rec[flat_points[i * n_profiles + j]] += weights[i]
    

# TODO
def reconstruct(xp: np.ndarray[np.int32],
                waterfall: np.ndarray[np.float64], n_iter: int,
                n_bins: int, n_particles: int, n_profiles: int,
                verbose: bool = ...,
                callback: Optional[object] = ...) -> tuple:
    ...