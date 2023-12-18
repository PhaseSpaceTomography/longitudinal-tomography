/**
 * @file reconstruct.h
 *
 * @author Anton Lu
 * Contact: anton.lu@cern.ch
 *
 * Functions in pure C/C++ that handles phase space reconstruction.
 * Meant to be called by a Python/C++ wrapper.
 */

#ifndef TOMO_RECONSTRUCT_H
#define TOMO_RECONSTRUCT_H

#include <functional>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include "pybind11/pybind11.h"

// Back projection using flattened arrays
template <typename T>
void back_project(T *weights,
                  int *flat_points,
                  const T *flat_profiles,
                  const int npart, const int nprof);

// Projections using flattened arrays
template <typename T>
void project(T *flat_rec,
             int *flat_points,
             const T *weights,
             const int npart, const int nprof);

template <typename T>
void normalize(T *flat_rec,
               const int nprof,
               const int nbins);

template <typename T>
void clip(T *array,
          const int length,
          const double clip_val);

template <typename T>
void find_difference_profile(T *diff_prof,
                             const T *flat_rec,
                             const T *flat_profiles,
                             const int all_bins);

template <typename T>
T discrepancy(const T *diff_prof,
              const int nprof,
              const int nbins);

template <typename T>
void compensate_particle_amount(T *diff_prof,
                                T *rparts,
                                const int nprof,
                                const int nbins);

template <typename T>
T max_2d(T **arr,
         const int x_axis,
         const int y_axis);

template <typename T>
T max_1d(T *arr, const int length);

template <typename T>
void count_particles_in_bin(T *rparts,
                            const int *xp,
                            const int nprof,
                            const int npart,
                            const int nbins);

template <typename T>
void reciprocal_particles(T *rparts,
                          const int *xp,
                          const int nbins,
                          const int nprof,
                          const int npart);

void create_flat_points(const int *xp,
                        int *flat_points,
                        const int npart,
                        const int nprof,
                        const int nbins);

template <typename T>
void reconstruct(T *weights,
                 const int *xp,
                 const T *flat_profiles,
                 T *flat_rec,
                 T *discr,
                 const int niter,
                 const int nbins,
                 const int npart,
                 const int nprof,
                 const bool verbose,
                 const std::function<void(int, int)> callback = 0);

#endif // TOMO_RECONSTRUCT_H
