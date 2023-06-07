"""Module containing the reconstruction algorithm with CUDA kernels.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
import cupy as cp
from ..utils import gpu_dev

if gpu_dev is None:
        from ..utils import GPUDev
        gpu_dev = GPUDev()

back_project_kernel = gpu_dev.rec_mod.get_function("back_project")
project_kernel = gpu_dev.rec_mod.get_function("project")
clip_kernel = gpu_dev.rec_mod.get_function("clip")
find_diffprof_kernel = gpu_dev.rec_mod.get_function("find_difference_profile")
count_part_bin_kernel = gpu_dev.rec_mod.get_function("count_particles_in_bin")
calc_reciprocal_kernel = gpu_dev.rec_mod.get_function("calculate_reciprocal")
comp_part_amount_kernel = gpu_dev.rec_mod.get_function("compensate_particle_amount")
create_flat_points_kernel = gpu_dev.rec_mod.get_function("create_flat_points")

block_size = gpu_dev.block_size
grid_size = gpu_dev.grid_size

def back_project(weights: cp.ndarray,
                 flat_points: cp.ndarray,
                 flat_profiles: cp.ndarray,
                 n_particles: int,
                 n_profiles: int) -> cp.ndarray:
    back_project_kernel(args=(weights, flat_points, flat_profiles, n_particles, n_profiles),
                        block=block_size,
                        grid=(int(n_particles / block_size[0] + 1), 1, 1))
    return weights

def project(flat_rec: cp.ndarray,
            flat_points: cp.ndarray,
            weights: cp.ndarray, n_particles: int,
            n_profiles: int, n_bins: int) -> cp.ndarray:
    project_kernel(args=(flat_rec, flat_points, weights, n_particles, n_profiles),
                        block=block_size,
                        grid=(int((n_particles * n_profiles) / block_size[0] + 1), 1, 1))
    return flat_rec

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
    array_length = len(array)
    clip_kernel(args=(array, array_length, clip_val),
                block=block_size,
                grid=(int(array_length / block_size[0] + 1), 1, 1))
    return array

def find_difference_profile(flat_rec: cp.ndarray,
                            flat_profiles: cp.ndarray) -> cp.ndarray:
    length = len(flat_rec)
    diff_prof = cp.empty(length)
    find_diffprof_kernel(args=(diff_prof, flat_rec, flat_profiles, length),
                         block=block_size,
                         grid=(int(length / block_size[0] + 1), 1, 1))
    return diff_prof

def discrepancy(diff_prof: cp.ndarray,
                n_profiles: int, n_bins: int) -> float:
    all_bins = n_profiles * n_bins
    squared_sum = cp.sum(cp.power(diff_prof, 2))

    return cp.sqrt(squared_sum / all_bins)

def compensate_particle_amount(diff_prof: cp.ndarray,
                               rparts: cp.ndarray,
                               n_profiles: int, n_bins: int) -> cp.ndarray:
    comp_part_amount_kernel(args=(diff_prof, rparts, n_profiles, n_bins),
                            block=block_size,
                            grid=(int((n_profiles * n_bins) / block_size[0] + 1), 1, 1))
    return diff_prof

def max_2d(array: cp.ndarray,
           x_axis: int, y_axis: int) -> float:
    return cp.max(array[:y_axis, :x_axis])

def reciprocal_particles(xp: cp.ndarray,
                         n_bins: int, n_profiles: int,
                         n_particles: int) -> cp.ndarray:
    rparts = cp.empty((n_profiles * n_bins), dtype=cp.float64)
    count_part_bin_kernel(args=(rparts, xp, n_profiles, n_particles, n_bins),
                          block=block_size,
                          grid=(int(n_particles / block_size[0] + 1), 1, 1))

    max_bin_val = float(cp.max(rparts))

    calc_reciprocal_kernel(args=(rparts, n_bins, n_profiles, max_bin_val),
                           block=block_size,
                           grid=(int((n_bins * n_profiles) / block_size[0] + 1), 1, 1))
    return rparts

def create_flat_points(xp: cp.ndarray,
                       n_particles: int, n_profiles: int,
                       n_bins: int) -> cp.ndarray:
    flat_points = cp.copy(xp)

    create_flat_points_kernel(args=(flat_points, n_particles, n_profiles, n_bins),
                              block=block_size,
                              grid=(int(n_particles / block_size[0] + 1), 1, 1))

    return flat_points

def reconstruct_cuda(xp: cp.ndarray,
                waterfall: cp.ndarray, n_iter: int,
                n_bins: int, n_particles: int, n_profiles: int,
                verbose: bool = ...) -> tuple:

    xp = xp.flatten()
    # from wrapper
    weights = cp.zeros(n_particles)
    discr = np.zeros(n_iter + 1)
    flat_profiles = waterfall.flatten()
    flat_rec = cp.zeros(n_profiles * n_bins)

    all_bins = n_profiles * n_bins
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