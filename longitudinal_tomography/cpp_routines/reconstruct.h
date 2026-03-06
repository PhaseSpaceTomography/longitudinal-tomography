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
template <typename real_int_t, typename int_t>
void back_project(real_int_t *weights,
                  int_t *flat_points,
                  const real_int_t *flat_profiles,
                  const int npart, const int nprof);

template <typename real_t>
void back_project_multi(real_t *weights,
                        int *flat_points,
                        const real_t *flat_profiles,
                        const bool *mask,
                        const int *centers,
                        const int npart,
                        const int nprof,
                        const int ncenter);

// Projections using flattened arrays
template <typename real_int_t, typename int_t>
void project(real_int_t *flat_rec,
             int_t *flat_points,
             const real_int_t *weights,
             const int npart, const int nprof);

template <typename real_t>
void project_multi(real_t *flat_rec,
                              int *flat_points,
                              const real_t *weights,
                              const int *centers,
                              const int npart,
                              const int nprof,
                              const int ncenter);

template <typename real_int_t, typename int_t>
void normalize(real_int_t *flat_rec,
               const int nprof,
               const int nbins,
               const int_t S = 1);

template <typename real_int_t>
void clip(real_int_t *array,
          const int length,
          const real_int_t clip_val);

template <typename real_int_t>
void find_difference_profile(real_int_t *diff_prof,
                             const real_int_t *flat_rec,
                             const real_int_t *flat_profiles,
                             const int all_bins);

template <typename real_int_t>
real_int_t discrepancy(const real_int_t *diff_prof,
              const int nprof,
              const int nbins);

template <typename real_t>
void discrepancy_multi(const real_t *diff_prof,
                         real_t *disc,
                         const int *cutleft,
                         const int *cutright,
                         const int iteration,
                         const int nprof,
                         const int nbins,
                         const int ncenter);


template <typename real_int_t>
void compensate_particle_amount(real_int_t *diff_prof,
                                real_int_t *rparts,
                                const int nprof,
                                const int nbins);

template <typename real_int_t>
real_int_t max_2d(real_int_t **arr,
         const int x_axis,
         const int y_axis);

template <typename real_int_t>
real_int_t max_1d(real_int_t *arr, const int length);

template <typename real_int_t, typename int_t>
void count_particles_in_bin(real_int_t *rparts,
                            const int_t *xp,
                            const int nprof,
                            const int npart,
                            const int nbins);

template <typename real_t>
void count_particles_in_bin_multi(real_t *rparts,
                                  const int *xpRound0,
                                  const int *centers,
                                  const int nprof,
                                  const int npart,
                                  const int nbins,
                                  const int ncenters);


template <typename real_int_t, typename int_t>
void reciprocal_particles(real_int_t *rparts,
                          const int_t *xp,
                          const int nbins,
                          const int nprof,
                          const int npart);

template <typename real_t>
void reciprocal_particles_multi(real_t *rparts,
                                const int *xpRound0,
                                const int *centers,
                                const int nbins,
                                const int nprof,
                                const int npart,
                                const int ncenters);

template <typename int_t>
void create_flat_points(const int_t *xp,
                        int_t *flat_points,
                        const int npart,
                        const int nprof,
                        const int nbins);

void create_mask(const int *xpRound0,
                 const int *centers,
                 const int *cutleft,
                 const int *cutright,
                 bool *mask,
                 const int npart,
                 const int nprof,
                 const int ncenter);

template <typename real_int_t, typename int_t>
void reconstruct(real_int_t *weights,
                 const int_t *xp,
                 const real_int_t *flat_profiles,
                 real_int_t *flat_rec,
                 real_int_t *discr,
                 const int niter,
                 const int nbins,
                 const int npart,
                 const int nprof,
                 const int_t S,
                 const bool verbose,
                 const std::function<void(int, int)> callback = 0);


template <typename real_t>
void reconstruct_multi(real_t *weights,
                       const int *xpRound0,
                       const int *centers,
                       const int *cutleft,
                       const int *cutright,
                       const real_t *flat_profiles,
                       real_t *flat_rec,
                       real_t *discr,
                       real_t *discr_split,
                       const int niter,
                       const int nbins,
                       const int npart,
                       const int nprof,
                       const int ncenter,
                       const bool verbose,
                       const std::function<void(int, int)> callback
);

#endif //TOMO_RECONSTRUCT_H