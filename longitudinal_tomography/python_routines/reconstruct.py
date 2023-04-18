"""Module containing the reconstruction algorithm derived from the cpp functions.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
from enum import Enum
from typing import Optional
from numba import njit, vectorize, prange
from ..cpp_routines import libtomo

class Mode(Enum):
    PURE = 1 # numpy
    JIT = 2
    JIT_PARALLEL = 3
    UNROLLED = 4
    UNROLLED_PARALLEL = 5
    VECTORIZE = 6
    VECTORIZE_PARALLEL = 7
    CPP = 8

# TODO create functions for different modes

# Probably no vectorization possible?
def back_project(weights: np.ndarray,
                 flat_points: np.ndarray,
                 flat_profiles: np.ndarray,
                 n_particles: int,
                 n_profiles: int) -> np.ndarray:
    for i in range(n_particles):
        for j in range(n_profiles):
            weights[i] += flat_profiles[flat_points[i, j]]
    return weights

def back_project_parallel(weights: np.ndarray,
                          flat_points: np.ndarray,
                          flat_profiles: np.ndarray,
                          n_particles: int,
                          n_profiles: int) -> np.ndarray:
    for i in prange(n_particles):
        for j in prange(n_profiles):
            weights[i] += flat_profiles[flat_points[i, j]]
    return weights

def project(flat_rec: np.ndarray,
            flat_points: np.ndarray,
            weights: np.ndarray, n_particles: int,
            n_profiles: int) -> np.ndarray:
    for i in range(n_particles):
        for j in range(n_profiles):
            flat_rec[flat_points[i, j]] += weights[i]
    return flat_rec

def project_parallel(flat_rec: np.ndarray,
                     flat_points: np.ndarray,
                     weights: np.ndarray, n_particles: int,
                     n_profiles: int) -> np.ndarray:
    for i in prange(n_particles):
        for j in prange(n_profiles):
            flat_rec[flat_points[i, j]] += weights[i]

def normalize(flat_rec: np.ndarray,
              n_profiles: int, n_bins: int) -> np.ndarray:
    sum_waterfall = 0.0
    for i in range(n_profiles):
        sum_profile = 0.0
        for j in range(n_bins):
            sum_profile += flat_rec[i * n_bins + j]
        for j in range(n_bins):
            flat_rec[i * n_bins + j] /= sum_profile
        sum_waterfall += sum_profile

    if sum_waterfall <= 0:
        raise RuntimeError("Phase space reduced to zeros!")
    return flat_rec

def normalize_parallel(flat_rec: np.ndarray,
                       n_profiles: int, n_bins: int) -> np.ndarray:
    sum_waterfall = 0.0
    for i in prange(n_profiles):
        sum_profile = 0.0
        for j in prange(n_bins):
            sum_profile += flat_rec[i * n_bins + j]
        for j in prange(n_bins):
            flat_rec[i * n_bins + j] /= sum_profile
        sum_waterfall += sum_profile

    if sum_waterfall <= 0:
        raise RuntimeError("Phase space reduced to zeros!")
    return flat_rec

def clip(array: np.ndarray,
        clip_val: float) -> np.ndarray:
    array[array < clip_val] = clip_val
    return array

def clip_unrolled(array: np.ndarray,
                  clip_val: float) -> np.ndarray:
    for i in range(array.shape[0]):
        if array[i] < clip_val:
            array[i] = clip_val
    return array

def clip_unrolled_parallel(array: np.ndarray,
                           clip_val: float) -> np.ndarray:
    for i in prange(array.shape[0]):
        if array[i] < clip_val:
            array[i] = clip_val
    return array

def clip_vectorized(arr_value: float, clip_val: float) -> float:
    if arr_value < clip_val:
        arr_value = clip_val
    return arr_value


def find_difference_profile(flat_rec: np.ndarray,
                            flat_profiles: np.ndarray) -> np.ndarray:
    return flat_profiles - flat_rec

def find_difference_profile_unrolled(flat_rec: np.ndarray,
                                     flat_profiles: np.ndarray) -> np.ndarray:
    diff_prof = np.zeros(flat_rec.shape)
    for i in range(flat_rec.shape[0]):
        diff_prof[i] = flat_profiles[i] - flat_rec[i]
    return diff_prof

def find_difference_profile_unrolled_parallel(flat_rec: np.ndarray,
                                              flat_profiles: np.ndarray) -> np.ndarray:
    diff_prof = np.zeros(flat_rec.shape)
    for i in prange(flat_rec.shape[0]):
        diff_prof[i] = flat_profiles[i] - flat_rec[i]
    return diff_prof

def find_difference_profile_vectorized(flat_rec: float, flat_profile: float) -> float:
    return flat_rec - flat_profile

def discrepancy(diff_prof: np.ndarray,
                n_profiles: int, n_bins: int) -> float:
    all_bins = n_profiles * n_bins
    squared_sum = np.sum(np.power(diff_prof, 2))

    return np.sqrt(squared_sum / all_bins)

def discrepancy_unrolled(diff_prof: np.ndarray,
                         n_profiles: int, n_bins: int) -> float:
    all_bins = n_profiles * n_bins
    squared_sum = 0
    for i in range(all_bins):
        squared_sum += diff_prof[i] ** 2
    return np.sqrt(squared_sum / all_bins)

def discrepancy_unrolled_parallel(diff_prof: np.ndarray,
                                  n_profiles: int, n_bins: int) -> float:
    all_bins = n_profiles * n_bins
    squared_sum = 0
    for i in prange(all_bins):
        squared_sum += diff_prof[i] ** 2
    return np.sqrt(squared_sum / all_bins)

def compensate_particle_amount(diff_prof: np.ndarray,
                               rparts: np.ndarray,
                               n_profiles: int, n_bins: int) -> np.ndarray:
    diff_prof *= rparts.reshape(n_profiles * n_bins)

    # for i in range(n_profiles):
    #     for j in range(n_bins):
    #         idx = i * n_bins + j
    #         diff_prof[idx] *= rparts[idx]

    return diff_prof

def compensate_particle_amount_unrolled(diff_prof: np.ndarray,
                                        rparts: np.ndarray,
                                        n_profiles: int, n_bins: int) -> np.ndarray:
    for i in range(n_profiles):
        for j in range(n_bins):
            idx = i * n_bins + j
            diff_prof[idx] *= rparts[idx]
    return diff_prof

def compensate_particle_amount_unrolled_parallel(diff_prof: np.ndarray,
                                        rparts: np.ndarray,
                                        n_profiles: int, n_bins: int) -> np.ndarray:
    for i in prange(n_profiles):
        for j in prange(n_bins):
            idx = i * n_bins + j
            diff_prof[idx] *= rparts[idx]
    return diff_prof


# VECTORIZATION POSSIBLE?

def max_2d(array: np.ndarray,
           x_axis: int, y_axis: int) -> float:
    return np.max(array[:y_axis, :x_axis])

def max_2d_unrolled(array: np.ndarray,
                    x_axis: int, y_axis: int) -> float:
    max_val = 0.0
    for i in range(y_axis):
        for j in range(x_axis):
            if(max_val < array[i][j]):
                max_val = array[i][j]
    return max_val

def max_2d_unrolled_parallel(array: np.ndarray,
                    x_axis: int, y_axis: int) -> float:
    max_val = 0.0
    for i in prange(y_axis):
        for j in prange(x_axis):
            if max_val < array[i][j]:
                max_val = array[i][j]
    return max_val

def max_1d(array: np.ndarray,
           length: int) -> float:
    return np.max(array)

def max_1d_unrolled(array: np.ndarray,
                    length: int) -> float:
    max_val = 0.0
    for i in range(length):
        if max_val < array[i]:
            max_val = array[i]
    return max_val

def max_1d_unrolled_parallel(array: np.ndarray,
                    length: int) -> float:
    max_val = 0.0
    for i in prange(length):
        if max_val < array[i]:
            max_val = array[i]
    return max_val

def count_particles_in_bin(xp: np.ndarray,
                           n_profiles: int, n_particles: int,
                           n_bins: int) -> np.ndarray:
    rparts = np.zeros((n_profiles, n_bins))

    for i in range(n_particles):
        for j in range(n_profiles):
            bin = xp[i, j]
            rparts[j, bin] += 1
    return rparts

def reciprocal_particles(xp: np.ndarray,
                         n_bins: int, n_profiles: int,
                         n_particles: int) -> np.ndarray:

    rparts = count_particles_in_bin(xp, n_profiles, n_particles, n_bins)
    max_bin_val = max_2d(rparts, n_particles, n_profiles)

    # Setting 0's to 1's to avoid zero division
    rparts = np.where(rparts == 0, 1, rparts)

    for i in range(n_profiles):
        for j in range(n_bins):
            #idx = i * n_bins + j
            rparts[i, j] = max_bin_val / rparts[i, j]
    return rparts

def create_flat_points(xp: np.ndarray,
                       n_particles: int, n_profiles: int,
                       n_bins: int) -> np.ndarray:
    flat_points = np.copy(xp)

    for i in range(n_particles):
        for j in range(n_profiles):
            flat_points[i, j] += n_bins * j
    return flat_points

def reconstruct(xp: np.ndarray,
                waterfall: np.ndarray, n_iter: int,
                n_bins: int, n_particles: int, n_profiles: int,
                verbose: bool = ...,
                callback: Optional[object] = ...) -> tuple:
    # from wrapper
    weights = np.zeros(n_particles)
    discr = np.zeros(n_iter + 1)
    flat_profiles = np.copy(waterfall.flatten()) # Shallow or deep copy needed?
    flat_rec = np.zeros(n_profiles * n_bins)

    all_bins = n_profiles * n_bins
    diff_prof = np.zeros(all_bins)
    flat_points = np.zeros(n_particles * n_profiles)

    # Actual functionality
    rparts = reciprocal_particles(xp, n_bins, n_profiles, n_particles)
    flat_points = create_flat_points(xp, n_particles, n_profiles, n_bins)
    weights = back_project(weights, flat_points, flat_profiles, n_particles, n_profiles)
    weights = clip(weights, 0.0)

    if np.sum(weights) <= 0:
        raise RuntimeError("All of phase space got reduced to zeros")

    if verbose:
        print(" Iterating...")

    for iteration in range(n_iter):
        if verbose:
            print(f"{iteration+1:3}")

        flat_rec = project(flat_rec, flat_points, weights, n_particles, n_profiles)
        flat_rec = normalize(flat_rec, n_profiles, n_bins)
        diff_prof = find_difference_profile(flat_rec, flat_profiles)
        discr[iteration] = discrepancy(diff_prof, n_profiles, n_bins)
        diff_prof = compensate_particle_amount(diff_prof, rparts, n_profiles, n_bins)
        weights = back_project(weights, flat_points, diff_prof, n_particles, n_profiles)
        weights = clip(weights, 0.0)

        if np.sum(weights) <= 0:
            raise RuntimeError("All of phase space got reduced to zeros")

        #callback(iteration + 1, n_iter)

    # Calculating final discrepancy
    flat_rec = project(flat_rec, flat_points, weights, n_particles, n_profiles)
    flat_rec = normalize(flat_rec, n_profiles, n_bins)
    diff_prof = find_difference_profile(flat_rec, flat_profiles)
    discr[n_iter] = discrepancy(diff_prof, n_profiles, n_bins)

    #callback(n_iter, n_iter)

    if verbose:
        print("Done!")

    return weights, discr, flat_rec