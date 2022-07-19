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
extern "C" void back_project(double *weights,
                             int *flat_points,
                             const double *flat_profiles,
                             const int npart, const int nprof);

extern "C" void back_project_multi(double *weights,
                                   int *flat_points,
                                   const double *flat_profiles,
                                   const bool *mask,
                                   const int *centers,
                                   const int npart,
                                   const int nprof,
                                   const int ncenter);

// Projections using flattened arrays
extern "C" void project(double *flat_rec,
                        int *flat_points,
                        const double *weights,
                        const int npart, const int nprof);


extern "C" void project_multi(double *flat_rec,
                              int *flat_points,
                              const double *weights,
                              const int *centers,
                              const int npart,
                              const int nprof,
                              const int ncenter);


void normalize(double *flat_rec,
               const int nprof,
               const int nbins);

void clip(double *array,
          const int length,
          const double clip_val);


void find_difference_profile(double *diff_prof,
                             const double *flat_rec,
                             const double *flat_profiles,
                             const int all_bins);

double discrepancy(const double *diff_prof,
                   const int nprof,
                   const int nbins);

void discrepancy_multi(const double *diff_prof,
                         double *disc,
                         const int *cutleft,
                         const int *cutright,
                         const int iteration,
                         const int nprof,
                         const int nbins,
                         const int ncenter);

void compensate_particle_amount(double *diff_prof,
                                double *rparts,
                                const int nprof,
                                const int nbins);

double max_2d(double **arr,
              const int x_axis,
              const int y_axis);

double max_1d(double *arr, const int length);

void count_particles_in_bin(double *rparts,
                            const int *xp,
                            const int nprof,
                            const int npart,
                            const int nbins);

void count_particles_in_bin_multi(double *rparts,
                                  const int *xpRound0,
                                  const int *centers,
                                  const int nprof,
                                  const int npart,
                                  const int nbins,
                                  const int ncenters);

void reciprocal_particles(double *rparts,
                          const int *xp,
                          const int nbins,
                          const int nprof,
                          const int npart);

void reciprocal_particles_multi(double *rparts,
                                const int *xpRound0,
                                const int *centers,
                                const int nbins,
                                const int nprof,
                                const int npart,
                                const int ncenters);

void create_flat_points(const int *xp,
                        int *flat_points,
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

extern "C" void reconstruct(double *weights,
                            const int *xp,
                            const double *flat_profiles,
                            double *flat_rec,
                            double *discr,
                            const int niter,
                            const int nbins,
                            const int npart,
                            const int nprof,
                            const bool verbose,
                            const std::function<void(int, int)> callback = 0);


extern "C" void reconstruct_multi(double *weights,
                                  const int *xpRound0,
                                  const int *centers,
                                  const int *cutleft,
                                  const int *cutright,
                                  const double *flat_profiles,
                                  double *flat_rec,
                                  double *discr,
                                  double *discr_split,
                                  const int niter,
                                  const int nbins,
                                  const int npart,
                                  const int nprof,
                                  const int ncenter,
                                  const bool verbose,
                                  const std::function<void(int, int)> callback
);

#endif //TOMO_RECONSTRUCT_H
