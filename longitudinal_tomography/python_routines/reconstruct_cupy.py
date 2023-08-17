"""Module containing the reconstruction algorithm derived from the cpp functions.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
import cupy as cp
import logging
from pyprof import timing

log = logging.getLogger(__name__)

# Probably no vectorization possible?
@timing.timeit(key='back_project')
def back_project(weights: cp.ndarray,
                 flat_points: cp.ndarray,
                 flat_profiles: cp.ndarray,
                 n_particles: int,
                 n_profiles: int) -> cp.ndarray:
    return cp.sum(cp.take(flat_profiles, flat_points, axis=0), axis=1) + weights

# @timing.timeit(key='back_project_jit')
# @jit.rawkernel()
# def back_project_jit(weights: cp.ndarray,
#                  flat_points: cp.ndarray,
#                  flat_profiles: cp.ndarray,
#                  n_particles: int,
#                  n_profiles: int) -> cp.ndarray:
#     BLOCK_SIZE = 32
#     ITEMS_PER_ARRAY = 16
#     ITEMS_PER_IT = BLOCK_SIZE * ITEMS_PER_ARRAY
#     iterations = (n_profiles + ITEMS_PER_IT - 1) // ITEMS_PER_IT

#     aggregate = cp.float64(0.0)
#     weight_prof = cp.zeros(16, dtype=cp.float64)
#     index = cp.int32(0)
#     fprof_index = cp.int32(0)
#     fpoints_index = cp.int32(0)
#     BlockReduce = jit.cub.BlockReduce[cp.float64, 32]
#     temp_storage = jit.shared_memory(
#         dtype=BlockReduce.TempStorage, size=1
#     )

#     for i in range(iterations):
#         # Using jit is experimental, using jit.cub.BlockReduce.Sum is even more
#         # because there is no documentation in CuPy available
#         # Reference: https://github.com/cupy/cupy/commit/95754ceabc7a3a1404a6fee5faa6e3f2ce120901
#         # Use at your own risk

#         #BlockReduce = jit.cub.BlockReduce[cp.float64, 32]
#         #temp_storage = jit.shared_memory(
#         #dtype=BlockReduce.TempStorage, size=1

#         for j in range(ITEMS_PER_ARRAY):
#             index = i * ITEMS_PER_IT + j * jit.blockDim.x + jit.threadIdx.x
#             fpoints_index = cp.int32(jit.blockIdx.x * n_profiles + index)
#             if index < n_profiles:
#                 fprof_index = int(flat_points[fpoints_index][0])
#                 weight_prof[j] = flat_profiles[fprof_index]
#         jit.syncthreads()

#         aggregate += BlockReduce(temp_storage[0]).Sum(weight_prof)

#     if jit.threadIdx.x == 0:
#         weights[jit.blockIdx.x] += aggregate


@timing.timeit(key='project')
def project(flat_rec: cp.ndarray,
            flat_points: cp.ndarray,
            weights: cp.ndarray, n_particles: int,
            n_profiles: int, n_bins: int) -> cp.ndarray:
    return cp.bincount(flat_points.reshape(-1), weights.repeat(n_profiles), minlength=n_profiles * n_bins) + flat_rec

def normalize(flat_rec: cp.ndarray,
              n_profiles: int, n_bins: int) -> cp.ndarray:

    flat_rec = flat_rec.reshape((n_profiles, n_bins))
    sum_profile = cp.sum(flat_rec, axis=1)
    flat_rec /= cp.expand_dims(sum_profile, axis=1)
    flat_rec = flat_rec.reshape(-1)
    sum_waterfall = cp.sum(sum_profile)

    if sum_waterfall <= 0:
        raise RuntimeError("Phase space reduced to zeros!")
    return flat_rec

def clip(array: cp.ndarray,
        clip_val: float) -> cp.ndarray:
    array[array < clip_val] = clip_val
    return array

def find_difference_profile(flat_rec: cp.ndarray,
                            flat_profiles: cp.ndarray) -> cp.ndarray:
    return flat_profiles - flat_rec

def discrepancy(diff_prof: cp.ndarray,
                n_profiles: int, n_bins: int) -> float:
    all_bins = n_profiles * n_bins
    squared_sum = cp.sum(cp.power(diff_prof, 2))

    return cp.sqrt(squared_sum / all_bins)

def compensate_particle_amount(diff_prof: cp.ndarray,
                               rparts: cp.ndarray,
                               n_profiles: int, n_bins: int) -> cp.ndarray:
    return diff_prof * rparts.reshape(n_profiles * n_bins)

def max_2d(array: cp.ndarray,
           x_axis: int, y_axis: int) -> float:
    return cp.max(array[:y_axis, :x_axis])

def count_particles_in_bin(xp: cp.ndarray,
                           n_profiles: int, n_particles: int,
                           n_bins: int) -> cp.ndarray:

    bins = cp.arange(n_bins + 1)
    rparts = cp.empty((n_profiles, n_bins), dtype=cp.int32)
    for j in range(n_profiles):
        rparts[j], _ = cp.histogram(xp[:, j], bins=bins)
    return rparts

    # return cp.sum(cp.eye(n_bins)[xp], axis=0) # extremely memory-intensive

def reciprocal_particles(xp: cp.ndarray,
                         n_bins: int, n_profiles: int,
                         n_particles: int) -> cp.ndarray:

    rparts = count_particles_in_bin(xp, n_profiles, n_particles, n_bins)
    max_bin_val = max_2d(rparts, n_particles, n_profiles)

    # Setting 0's to 1's to avoid zero division
    rparts = cp.where(rparts == 0, 1, rparts)
    rparts = max_bin_val / rparts

    return rparts

def create_flat_points(xp: cp.ndarray,
                       n_particles: int, n_profiles: int,
                       n_bins: int) -> cp.ndarray:
    flat_points = cp.copy(xp)
    flat_points += cp.arange(n_profiles) * n_bins

    return flat_points

def reconstruct_cupy(xp: cp.ndarray,
                waterfall: cp.ndarray, n_iter: int,
                n_bins: int, n_particles: int, n_profiles: int,
                verbose: bool = ...) -> tuple:

    # from wrapper
    weights = cp.zeros(n_particles)
    discr = np.zeros(n_iter + 1)
    flat_profiles = waterfall.flatten()
    flat_rec = cp.zeros(n_profiles * n_bins)

    all_bins = n_profiles * n_bins
    diff_prof = cp.zeros(all_bins)
    flat_points = cp.zeros(n_particles * n_profiles)

    # Actual functionality
    rparts = reciprocal_particles(xp, n_bins, n_profiles, n_particles)
    flat_points = create_flat_points(xp, n_particles, n_profiles, n_bins)
    weights = back_project(weights, flat_points, flat_profiles, n_particles, n_profiles)
    weights = clip(weights, 0.0)

    if cp.sum(weights) <= 0:
        raise RuntimeError("All of phase space got reduced to zeros")

    if verbose:
        print(" Iterating...")

    for iteration in range(n_iter):
        if verbose:
            print(f"{iteration+1:3}")

        flat_rec = project(flat_rec, flat_points, weights, n_particles, n_profiles, n_bins)
        flat_rec = normalize(flat_rec, n_profiles, n_bins)
        diff_prof = find_difference_profile(flat_rec, flat_profiles)
        discr[iteration] = discrepancy(diff_prof, n_profiles, n_bins)
        diff_prof = compensate_particle_amount(diff_prof, rparts, n_profiles, n_bins)
        weights = back_project(weights, flat_points, diff_prof, n_particles, n_profiles)
        weights = clip(weights, 0.0)


        if cp.sum(weights) <= 0:
            raise RuntimeError("All of phase space got reduced to zeros")

    # Calculating final discrepancy
    flat_rec = project(flat_rec, flat_points, weights, n_particles, n_profiles, n_bins)
    flat_rec = normalize(flat_rec, n_profiles, n_bins)
    diff_prof = find_difference_profile(flat_rec, flat_profiles)
    discr[n_iter] = discrepancy(diff_prof, n_profiles, n_bins)

    if verbose:
        print("Done!")

    return weights, discr, flat_rec