"""Module containing the reconstruction algorithm derived from the cpp functions.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
from typing import Optional
from numba import njit, prange

@njit(parallel=True)
def back_project(weights: np.ndarray,
                 flat_points: np.ndarray,
                 flat_profiles: np.ndarray,
                 n_particles: int,
                 n_profiles: int) -> np.ndarray:
    for i in prange(n_particles):
        for j in prange(n_profiles):
            weights[i] += flat_profiles[flat_points[i * n_profiles + j]]
    return weights

@njit(parallel=True)
def project(flat_rec: np.ndarray,
            flat_points: np.ndarray,
            weights: np.ndarray, n_particles: int,
            n_profiles: int) -> np.ndarray:
    for i in prange(n_particles):
        for j in prange(n_profiles):
            flat_rec[flat_points[i * n_profiles + j]] += weights[i]
    return flat_rec

@njit(parallel=True)
def normalize(flat_rec: np.ndarray,
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

@njit(parallel=True)
def clip(array: np.ndarray,
        clip_val: float) -> np.ndarray:
    array[array < clip_val] = clip_val
    return array

@njit(parallel=True)
def find_difference_profile(flat_rec: np.ndarray,
                            flat_profiles: np.ndarray) -> np.ndarray:
    return flat_profiles - flat_rec

@njit(parallel=True)
def discrepancy(diff_prof: np.ndarray,
                n_profiles: int, n_bins = int) -> float:
    all_bins = n_profiles * n_bins
    squared_sum = np.sum(np.power(diff_prof, 2))

    return np.sqrt(squared_sum / all_bins)

@njit(parallel=True)
def compensate_particle_amount(diff_prof: np.ndarray,
                               rparts: np.ndarray,
                               n_profiles: int, n_bins: int) -> np.ndarray:
    diff_prof *= rparts.reshape(n_profiles * n_bins)

    # for i in prange(n_profiles):
    #     for j in prange(n_bins):
    #         idx = i * n_bins + j
    #         diff_prof[idx] *= rparts[idx]

    return diff_prof

@njit(parallel=True)
def max_2d(array: np.ndarray,
           x_axis: int, y_axis: int) -> float:
    return np.max(array[:y_axis, :x_axis])

@njit(parallel=True)
def count_particles_in_bin(xp: np.ndarray,
                           n_profiles: int, n_particles: int,
                           n_bins: int) -> np.ndarray:
    rparts = np.zeros((n_profiles, n_bins))

    for i in prange(n_particles):
        for j in prange(n_profiles):
            bin = xp[i, j]
            rparts[j, bin] += 1
    return rparts

@njit(parallel=True)
def reciprocal_particles(xp: np.ndarray,
                         n_bins: int, n_profiles: int,
                         n_particles: int) -> np.ndarray:
    all_bins = n_profiles * n_bins
    rparts = count_particles_in_bin(xp, n_profiles, n_particles, n_bins)
    max_bin_val = max_2d(rparts, n_particles, n_profiles)

    # Setting 0's to 1's to avoid zero division
    rparts = np.where(rparts == 0, 1, rparts)

    for i in prange(n_profiles):
        for j in prange(n_bins):
            #idx = i * n_bins + j
            rparts[i, j] = max_bin_val / rparts[i, j]
    return rparts

@njit(parallel=True)
def create_flat_points(xp: np.ndarray,
                       n_particles: int, n_profiles: int,
                       n_bins: int) -> np.ndarray:
    flat_points = np.copy(xp.flatten())

    for i in prange(n_particles):
        for j in prange(n_profiles):
            flat_points[i * n_profiles + j] += n_bins * j
    return flat_points

def reconstruct_numba(xp: np.ndarray,
                waterfall: np.ndarray, n_iter: int,
                n_bins: int, n_particles: int, n_profiles: int,
                verbose: bool) -> tuple:
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

    for iteration in prange(n_iter):
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