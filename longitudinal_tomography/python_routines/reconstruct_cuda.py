"""Module containing the reconstruction algorithm with CUDA kernels.

:Author(s): **Bernardo Abreu Figueiredo**
"""
from __future__ import annotations

import numpy as np
import cupy as cp
from typing import TYPE_CHECKING
from ..utils.tomo_config import GPUDev
from ..utils.tomo_config import AppConfig as conf
import longitudinal_tomography.cuda_kernels as cuda_kernels

if TYPE_CHECKING:
    from cupy.typing import NDArray as CPArray

gpu_dev = GPUDev.get_gpu_dev()
block_size = gpu_dev.block_size

def refresh_kernels():
    global back_project_kernel, project_kernel, clip_kernel, find_diffprof_kernel,\
        count_part_bin_kernel, calc_reciprocal_kernel, comp_part_amount_kernel, create_flat_points_kernel

    gpu_dev = GPUDev.get_gpu_dev()
    back_project_kernel = gpu_dev.rec_mod.get_function("back_project")
    project_kernel = gpu_dev.rec_mod.get_function("project")
    clip_kernel = gpu_dev.rec_mod.get_function("clip")
    find_diffprof_kernel = gpu_dev.rec_mod.get_function("find_difference_profile")
    count_part_bin_kernel = gpu_dev.rec_mod.get_function("count_particles_in_bin")
    calc_reciprocal_kernel = gpu_dev.rec_mod.get_function("calculate_reciprocal")
    comp_part_amount_kernel = gpu_dev.rec_mod.get_function("compensate_particle_amount")
    create_flat_points_kernel = gpu_dev.rec_mod.get_function("create_flat_points")


def back_project(weights: CPArray,
                 flat_points: CPArray,
                 flat_profiles: CPArray,
                 n_particles: int,
                 n_profiles: int) -> CPArray:
    back_project_kernel(args=(weights, flat_points, flat_profiles, n_particles, n_profiles),
                            block=(cuda_kernels.REDUCTION_BLOCK_SIZE, 1, 1),
                            grid=(n_particles, 1, 1))
    return weights


def project(flat_rec: CPArray,
            flat_points: CPArray,
            weights: CPArray, n_particles: int,
            n_profiles: int, n_bins: int) -> CPArray:
    project_kernel(args=(flat_rec, flat_points, weights, n_particles, n_profiles),
                        block=block_size,
                        grid=(int((n_particles * n_profiles) / block_size[0] + 1), 1, 1))
    return flat_rec

def normalize(flat_rec: CPArray,
              n_profiles: int, n_bins: int) -> CPArray:
    flat_rec = flat_rec.reshape((n_profiles, n_bins))
    sum_profile = cp.sum(flat_rec, axis=1)
    flat_rec /= cp.expand_dims(sum_profile, axis=1)
    flat_rec = flat_rec.reshape(-1)
    sum_waterfall = cp.sum(sum_profile)

    if sum_waterfall <= 0:
        raise RuntimeError("Phase space reduced to zeros!")
    return flat_rec

def clip(array: CPArray,
         array_length: int,
        clip_val: float) -> CPArray:
    clip_kernel(args=(array, array_length, clip_val),
                block=block_size,
                grid=(int(array_length / block_size[0] + 1), 1, 1))
    return array

def find_difference_profile(flat_rec: CPArray,
                            flat_profiles: CPArray) -> CPArray:
    length = len(flat_rec)
    diff_prof = cp.empty(length, dtype=flat_rec.dtype)
    find_diffprof_kernel(args=(diff_prof, flat_rec, flat_profiles, length),
                         block=block_size,
                         grid=(int(length / block_size[0] + 1), 1, 1))
    return diff_prof

def discrepancy(diff_prof: CPArray,
                n_profiles: int, n_bins: int) -> float:
    all_bins = n_profiles * n_bins
    squared_sum = cp.sum(cp.power(diff_prof, 2))

    return cp.sqrt(squared_sum / all_bins)

def compensate_particle_amount(diff_prof: CPArray,
                               rparts: CPArray,
                               n_profiles: int, n_bins: int) -> CPArray:
    comp_part_amount_kernel(args=(diff_prof, rparts, n_profiles, n_bins),
                            block=block_size,
                            grid=(int((n_profiles * n_bins) / block_size[0] + 1), 1, 1))
    return diff_prof

def max_2d(array: CPArray,
           x_axis: int, y_axis: int) -> float:
    return cp.max(array[:y_axis, :x_axis])

def reciprocal_particles(rparts: CPArray, xp: CPArray,
                         n_bins: int, n_profiles: int,
                         n_particles: int) -> CPArray:
    count_part_bin_kernel(args=(rparts, xp, n_profiles, n_particles, n_bins),
                          block=block_size,
                          grid=(int((n_particles * n_profiles) / block_size[0] + 1), 1, 1))

    max_bin_val = float(cp.max(rparts))

    calc_reciprocal_kernel(args=(rparts, n_bins, n_profiles, max_bin_val),
                           block=block_size,
                           grid=(int((n_bins * n_profiles) / block_size[0] + 1), 1, 1))

    return rparts

def create_flat_points(xp: CPArray,
                       n_particles: int, n_profiles: int,
                       n_bins: int) -> CPArray:
    flat_points = cp.copy(xp)

    create_flat_points_kernel(args=(flat_points, n_particles, n_profiles, n_bins),
                              block=block_size,
                              grid=(int((n_particles * n_profiles) / block_size[0] + 1), 1, 1))

    return flat_points

def reconstruct_cuda(xp: CPArray,
                waterfall: CPArray, n_iter: int,
                n_bins: int, n_particles: int, n_profiles: int,
                verbose: bool = False, callback = None) -> tuple:
    xp = xp.flatten()
    # from wrapper
    weights = cp.zeros(n_particles, dtype=conf.get_precision())
    discr = np.zeros(n_iter + 1)
    flat_profiles = waterfall.flatten().astype(conf.get_precision())
    flat_rec = cp.zeros(n_profiles * n_bins, dtype=conf.get_precision())
    flat_points = cp.zeros(n_particles * n_profiles)
    rparts = cp.zeros((n_profiles * n_bins), dtype=conf.get_precision())

    # Actual functionality
    rparts = reciprocal_particles(rparts, xp, n_bins, n_profiles, n_particles)
    flat_points = create_flat_points(xp, n_particles, n_profiles, n_bins)
    weights = back_project(weights, flat_points, flat_profiles, n_particles, n_profiles)
    weights = clip(weights, n_particles, 0.0)

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
        weights = clip(weights, n_particles, 0.0)

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

refresh_kernels()