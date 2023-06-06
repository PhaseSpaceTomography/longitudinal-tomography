"""Module containing the reconstruction algorithm derived from the cpp functions.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
from numba import njit, vectorize, prange
from ..cpp_routines import libtomo
from ..utils.execution_mode import Mode

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

def back_project_cpp(weights: np.ndarray,
                 flat_points: np.ndarray,
                 flat_profiles: np.ndarray,
                 n_particles: int,
                 n_profiles: int) -> np.ndarray:
    return libtomo.back_project(weights, flat_points, flat_profiles, n_particles, n_profiles)

def project(flat_rec: np.ndarray,
            flat_points: np.ndarray,
            weights: np.ndarray, n_particles: int,
            n_profiles: int, n_bins: int) -> np.ndarray:
    for i in range(n_particles):
        for j in range(n_profiles):
            flat_rec[flat_points[i, j]] += weights[i]
    return flat_rec

def project_parallel(flat_rec: np.ndarray,
                     flat_points: np.ndarray,
                     weights: np.ndarray, n_particles: int,
                     n_profiles: int, n_bins: int) -> np.ndarray:
    for i in prange(n_particles):
        for j in prange(n_profiles):
            flat_rec[flat_points[i, j]] += weights[i]
    return flat_rec

def project_cpp(flat_rec: np.ndarray,
            flat_points: np.ndarray,
            weights: np.ndarray, n_particles: int,
            n_profiles: int, n_bins: int) -> np.ndarray:
    return libtomo.project(flat_rec, flat_points, weights, n_particles, n_profiles, n_bins)

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

def normalize_cpp(flat_rec: np.ndarray,
                  n_profiles: int, n_bins: int) -> np.ndarray:
    return libtomo.normalize(flat_rec, n_profiles, n_bins)

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

def clip_cpp(array: np.ndarray, clip_val: float) -> np.ndarray:
    return libtomo.clip(array, len(array), clip_val)


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
    diff_prof = flat_profile - flat_rec
    return diff_prof

def find_difference_profile_cpp(flat_rec: np.ndarray,
                                flat_profiles: np.ndarray) -> np.ndarray:
    diff_prof = np.zeros(len(flat_rec))
    return libtomo.find_difference_profile(diff_prof, flat_rec, flat_profiles, len(flat_rec))

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

def discrepancy_cpp(diff_prof: np.ndarray,
                    n_profiles: int, n_bins: int) -> float:
    return libtomo.discrepancy(diff_prof, n_profiles, n_bins)

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
            diff_prof[idx] *= rparts[i, j]
    return diff_prof

def compensate_particle_amount_unrolled_parallel(diff_prof: np.ndarray,
                                        rparts: np.ndarray,
                                        n_profiles: int, n_bins: int) -> np.ndarray:
    for i in prange(n_profiles):
        for j in prange(n_bins):
            idx = i * n_bins + j
            diff_prof[idx] *= rparts[i, j]
    return diff_prof

def compensate_particle_amount_cpp(diff_prof: np.ndarray,
                                   rparts: np.ndarray,
                                   n_profiles: int, n_bins: int) -> np.ndarray:
    return libtomo.compensate_particle_amount(diff_prof, rparts, n_profiles, n_bins)

@njit
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

def max_2d_cpp(array: np.ndarray,
               x_axis: int, y_axis: int) -> float:
    return libtomo.max_2d(array, x_axis, y_axis)

@njit
def count_particles_in_bin(xp: np.ndarray,
                           n_profiles: int, n_particles: int,
                           n_bins: int) -> np.ndarray:
    rparts = np.zeros((n_profiles, n_bins))

    for i in range(n_particles):
        for j in range(n_profiles):
            index = xp[i, j]
            rparts[j, index] += 1
    return rparts

@njit(parallel=True)
def count_particles_in_bin_parallel(xp: np.ndarray,
                                    n_profiles: int, n_particles: int,
                                    n_bins: int) -> np.ndarray:
    rparts = np.zeros((n_profiles, n_bins))

    for i in prange(n_particles):
        for j in prange(n_profiles):
            index = xp[i, j]
            rparts[j, index] += 1
    return rparts

def count_particles_in_bin_cpp(xp: np.ndarray,
                               n_profiles: int, n_particles: int,
                               n_bins: int) -> np.ndarray:
    rparts = np.zeros(n_profiles * n_bins)
    return libtomo.count_particles_in_bin(rparts, xp, n_profiles, n_particles, n_bins)

def reciprocal_particles(xp: np.ndarray,
                         n_bins: int, n_profiles: int,
                         n_particles: int) -> np.ndarray:

    rparts = count_particles_in_bin(xp, n_profiles, n_particles, n_bins)
    max_bin_val = max_2d(rparts, n_particles, n_profiles)

    # Setting 0's to 1's to avoid zero division
    rparts = np.where(rparts == 0, 1, rparts)

    for i in range(n_profiles):
        for j in range(n_bins):
            rparts[i, j] = max_bin_val / rparts[i, j]
    return rparts

def reciprocal_particles_parallel(xp: np.ndarray,
                         n_bins: int, n_profiles: int,
                         n_particles: int) -> np.ndarray:

    rparts = count_particles_in_bin_parallel(xp, n_profiles, n_particles, n_bins)
    max_bin_val = max_2d(rparts, n_particles, n_profiles)

    # Setting 0's to 1's to avoid zero division
    rparts = np.where(rparts == 0, 1, rparts)

    for i in prange(n_profiles):
        for j in prange(n_bins):
            rparts[i, j] = max_bin_val / rparts[i, j]
    return rparts

def reciprocal_particles_cpp(xp: np.ndarray,
                             n_bins: int, n_profiles: int,
                             n_particles: int) -> np.ndarray:
    rparts = np.zeros(n_profiles * n_bins)
    return libtomo.reciprocal_particles(rparts, xp, n_bins, n_profiles, n_particles)

def create_flat_points(xp: np.ndarray,
                       n_particles: int, n_profiles: int,
                       n_bins: int) -> np.ndarray:
    flat_points = np.copy(xp)

    for i in range(n_particles):
        for j in range(n_profiles):
            flat_points[i, j] += n_bins * j
    return flat_points

def create_flat_points_parallel(xp: np.ndarray,
                       n_particles: int, n_profiles: int,
                       n_bins: int) -> np.ndarray:
    flat_points = np.copy(xp)

    for i in prange(n_particles):
        for j in prange(n_profiles):
            flat_points[i, j] += n_bins * j
    return flat_points

def create_flat_points_cpp(xp: np.ndarray,
                           n_particles: int, n_profiles: int,
                           n_bins: int) -> np.ndarray:
    flat_points = np.zeros(n_particles * n_profiles)
    return libtomo.create_flat_points(xp, flat_points, n_particles, n_profiles, n_bins)

def reconstruct(xp: np.ndarray,
                waterfall: np.ndarray, n_iter: int,
                n_bins: int, n_particles: int, n_profiles: int,
                verbose: bool = ...,
                mode: Mode = Mode.JIT) -> tuple:

    if mode == Mode.JIT_PARALLEL:
        from .reconstruct_numba import reconstruct_numba
        return reconstruct_numba(xp, waterfall, n_iter, n_bins, n_particles, n_profiles, verbose)

    # TODO remove this
    # functions
    reciprocal_particles_func = reciprocal_particles
    create_flat_points_func = create_flat_points
    back_project_func = back_project
    clip_func = clip
    project_func = project
    normalize_func = normalize
    find_difference_profile_func = find_difference_profile
    discrepancy_func = discrepancy
    compensate_particle_amount_func = compensate_particle_amount

    if mode == Mode.JIT or mode == Mode.VECTORIZE or mode == Mode.CUPY:
        reciprocal_particles_func = njit()(reciprocal_particles)
        create_flat_points_func = njit()(create_flat_points)
        back_project_func = njit()(back_project)
        clip_func = njit()(clip) if mode == Mode.JIT or mode == Mode.CUPY \
            else vectorize()(clip_vectorized)
        project_func = njit()(project)
        normalize_func = njit()(normalize)
        find_difference_profile_func = njit()(find_difference_profile) if mode == Mode.JIT or mode == Mode.CUPY \
            else vectorize()(find_difference_profile_vectorized)
        discrepancy_func = njit()(discrepancy)
        compensate_particle_amount_func = njit()(compensate_particle_amount)
    elif mode == Mode.JIT_PARALLEL or mode == Mode.VECTORIZE_PARALLEL:
        reciprocal_particles_func = njit(parallel=True)(reciprocal_particles_parallel)
        create_flat_points_func = njit(parallel=True)(create_flat_points_parallel)
        back_project_func = njit(parallel=True)(back_project_parallel)
        clip_func = njit(parallel=True)(clip) if mode == Mode.JIT_PARALLEL \
            else vectorize('float64(float64, float64)', target='parallel')(clip_vectorized)
        project_func = njit(parallel=True)(project_parallel)
        normalize_func = njit(parallel=True)(normalize_parallel)
        find_difference_profile_func = njit(parallel=True)(find_difference_profile) if mode == Mode.JIT_PARALLEL \
            else vectorize('float64(float64, float64)', target='parallel')(find_difference_profile_vectorized)
        discrepancy_func = njit(parallel=True)(discrepancy)
        compensate_particle_amount_func = njit(parallel=True)(compensate_particle_amount)
    elif mode == Mode.UNROLLED:
        reciprocal_particles_func = njit()(reciprocal_particles)
        create_flat_points_func = njit()(create_flat_points)
        back_project_func = njit()(back_project)
        clip_func = njit()(clip)
        project_func = njit()(project)
        normalize_func = njit()(normalize)
        find_difference_profile_func = njit()(find_difference_profile)
        discrepancy_func = njit()(discrepancy)
        compensate_particle_amount_func = njit()(compensate_particle_amount)
    elif mode == Mode.UNROLLED_PARALLEL:
        reciprocal_particles_func = njit(parallel=True)(reciprocal_particles_parallel)
        create_flat_points_func = njit(parallel=True)(create_flat_points_parallel)
        back_project_func = njit(parallel=True)(back_project_parallel)
        clip_func = njit(parallel=True)(clip_unrolled_parallel)
        project_func = njit(parallel=True)(project_parallel)
        normalize_func = njit(parallel=True)(normalize_parallel)
        find_difference_profile_func = njit(parallel=True)(find_difference_profile_unrolled_parallel)
        discrepancy_func = njit(parallel=True)(discrepancy_unrolled_parallel)
        compensate_particle_amount_func = njit(parallel=True)(compensate_particle_amount_unrolled_parallel)
    elif mode == Mode.CPP_WRAPPER:
        reciprocal_particles_func = reciprocal_particles_cpp
        create_flat_points_func = create_flat_points_cpp
        back_project_func = back_project_cpp
        clip_func = clip_cpp
        project_func = project_cpp
        normalize_func = normalize_cpp
        find_difference_profile_func = find_difference_profile_cpp
        discrepancy_func = discrepancy_cpp
        compensate_particle_amount_func = compensate_particle_amount_cpp


    # from wrapper
    weights = np.zeros(n_particles)
    discr = np.zeros(n_iter + 1)
    flat_profiles = np.copy(waterfall.flatten())
    flat_rec = np.zeros(n_profiles * n_bins)

    all_bins = n_profiles * n_bins
    diff_prof = np.zeros(all_bins)
    flat_points = np.zeros(n_particles * n_profiles)

    # Actual functionality
    rparts = reciprocal_particles_func(xp, n_bins, n_profiles, n_particles)
    flat_points = create_flat_points_func(xp, n_particles, n_profiles, n_bins)
    weights = back_project_func(weights, flat_points, flat_profiles, n_particles, n_profiles)
    weights = clip_func(weights, 0.0)

    if np.sum(weights) <= 0:
        raise RuntimeError("All of phase space got reduced to zeros")

    if verbose:
        print(" Iterating...")

    for iteration in range(n_iter):
        if verbose:
            print(f"{iteration+1:3}")

        flat_rec = project_func(flat_rec, flat_points, weights, n_particles, n_profiles, n_bins)
        flat_rec = normalize_func(flat_rec, n_profiles, n_bins)
        diff_prof = find_difference_profile_func(flat_rec, flat_profiles)
        discr[iteration] = discrepancy_func(diff_prof, n_profiles, n_bins)
        diff_prof = compensate_particle_amount_func(diff_prof, rparts, n_profiles, n_bins)
        weights = back_project_func(weights, flat_points, diff_prof, n_particles, n_profiles)
        weights = clip_func(weights, 0.0)

        if np.sum(weights) <= 0:
            raise RuntimeError("All of phase space got reduced to zeros")

    # Calculating final discrepancy
    flat_rec = project_func(flat_rec, flat_points, weights, n_particles, n_profiles, n_bins)
    flat_rec = normalize_func(flat_rec, n_profiles, n_bins)
    diff_prof = find_difference_profile_func(flat_rec, flat_profiles)
    discr[n_iter] = discrepancy_func(diff_prof, n_profiles, n_bins)

    if verbose:
        print("Done!")

    return weights, discr, flat_rec