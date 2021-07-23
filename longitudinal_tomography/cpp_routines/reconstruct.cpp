/**
 * @file reconstruct.cpp
 *
 * @author Anton Lu
 * Contact: anton.lu@cern.ch
 *
 * Functions in pure C/C++ that handles phase space reconstruction.
 * Meant to be called by a Python/C++ wrapper.
 */

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <functional>

#include "reconstruct.h"

// Back projection using flattened arrays
extern "C" void back_project(double *weights,                     // inn/out
                             int *flat_points,       // inn
                             const double *flat_profiles,         // inn
                             const int npart, const int nprof) {     // inn
#pragma omp parallel for
    for (int i = 0; i < npart; i++)
        for (int j = 0; j < nprof; j++)
            weights[i] += flat_profiles[flat_points[i * nprof + j]];
}

// Projections using flattened arrays
extern "C" void project(double *flat_rec,                     // inn/out
                        int *flat_points,        // inn
                        const double *weights,   // inn
                        const int npart, const int nprof) {      // inn
    for (int i = 0; i < npart; i++)
        for (int j = 0; j < nprof; j++)
            flat_rec[flat_points[i * nprof + j]] += weights[i];
}

void normalize(double *flat_rec, // inn/out
               const int nprof,
               const int nbins) {
    double sum_waterfall = 0.0;
#pragma omp parallel for reduction(+ : sum_waterfall)
    for (int i = 0; i < nprof; i++) {
        double sum_profile = 0;
        for (int j = 0; j < nbins; j++)
            sum_profile += flat_rec[i * nbins + j];
        for (int j = 0; j < nbins; j++)
            flat_rec[i * nbins + j] /= sum_profile;
        sum_waterfall += sum_profile;
    }

    if (sum_waterfall <= 0)
        throw std::runtime_error("Phase space reduced to zeroes!");
}

void clip(double *array, // inn/out
          const int length,
          const double clip_val) {
#pragma omp parallel for
    for (int i = 0; i < length; i++)
        if (array[i] < clip_val)
            array[i] = clip_val;
}


void find_difference_profile(double *diff_prof,           // out
                             const double *flat_rec,      // inn
                             const double *flat_profiles, // inn
                             const int all_bins) {
#pragma omp parallel for
    for (int i = 0; i < all_bins; i++)
        diff_prof[i] = flat_profiles[i] - flat_rec[i];
}

double discrepancy(const double *diff_prof,   // inn
                   const int nprof,
                   const int nbins) {
    int all_bins = nprof * nbins;
    double squared_sum = 0;

    for (int i = 0; i < all_bins; i++) {
        squared_sum += std::pow(diff_prof[i], 2.0);
    }

    return std::sqrt(squared_sum / (nprof * nbins));
}

void compensate_particle_amount(double *diff_prof,        // inn/out
                                double *rparts,          // inn
                                const int nprof,
                                const int nbins) {
#pragma omp parallel for
    for (int i = 0; i < nprof; i++)
        for (int j = 0; j < nbins; j++) {
            int idx = i * nbins + j;
            diff_prof[idx] *= rparts[idx];
        }
}

double max_2d(double **arr,  // inn
              const int x_axis,
              const int y_axis) {
    double max_bin_val = 0;
    for (int i = 0; i < y_axis; i++)
        for (int j = 0; j < x_axis; j++)
            if (max_bin_val < arr[i][j])
                max_bin_val = arr[i][j];
    return max_bin_val;
}

double max_1d(double *arr, const int length) {
    double max_bin_val = 0;
    for (int i = 0; i < length; i++)
        if (max_bin_val < arr[i])
            max_bin_val = arr[i];
    return max_bin_val;
}


double sum(double *arr, const int length) {
    double sum = 0;
    for (int i = 0; i < length; i++)
        sum += arr[i];
    return sum;
}

void count_particles_in_bin(double *rparts,      // out
                            const int *xp,       // inn
                            const int nprof,
                            const int npart,
                            const int nbins) {
    int bin;
    for (int i = 0; i < npart; i++)
        for (int j = 0; j < nprof; j++) {
            bin = xp[i * nprof + j];
            rparts[j * nbins + bin] += 1;
        }
}

void reciprocal_particles(double *rparts,   // out
                          const int *xp,     // inn
                          const int nbins,
                          const int nprof,
                          const int npart) {
    const int all_bins = nprof * nbins;

    count_particles_in_bin(rparts, xp, nprof, npart, nbins);

    int max_bin_val = max_1d(rparts, all_bins);

    // Setting 0's to 1's to avoid zero division
#pragma omp parallel for
    for (int i = 0; i < all_bins; i++)
        if (rparts[i] == 0.0)
            rparts[i] = 1.0;

    // Creating reciprocal
    int idx;
#pragma omp parallel for
    for (int i = 0; i < nprof; i++)
        for (int j = 0; j < nbins; j++) {
            idx = i * nbins + j;
            rparts[idx] = (double) max_bin_val / rparts[idx];
        }
}

void create_flat_points(const int *xp,       //inn
                        int *flat_points,    //out
                        const int npart,
                        const int nprof,
                        const int nbins) {
    // Initiating to the value of xp
    std::memcpy(flat_points, xp, npart * nprof * sizeof(int));

    for (int i = 0; i < npart; i++)
        for (int j = 0; j < nprof; j++)
            flat_points[i * nprof + j] += nbins * j;
}


extern "C" void reconstruct(double *weights,             // out
                            const int *xp,              // inn
                            const double *flat_profiles, // inn
                            double *flat_rec,            // Out
                            double *discr,               // out
                            const int niter,
                            const int nbins,
                            const int npart,
                            const int nprof,
                            const bool verbose,
                            const std::function<void(int, int)> callback
) {
    // Creating arrays...
    int all_bins = nprof * nbins;
    double *diff_prof = new double[all_bins]();

    double *rparts = new double[all_bins]();

    int *flat_points = new int[npart * nprof]();

    auto cleanup = [diff_prof, flat_points, rparts]() {
        delete[] diff_prof;
        delete[] rparts;
        delete[] flat_points;
    };

    // Actual functionality

    try {
        reciprocal_particles(rparts, xp, nbins, nprof, npart);

        create_flat_points(xp, flat_points, npart, nprof, nbins);

        back_project(weights, flat_points, flat_profiles, npart, nprof);
        clip(weights, npart, 0.0);

        if (sum(weights, npart) <= 0.)
            throw std::runtime_error("All of phase space got reduced to zeroes");

        if (verbose)
            std::cout << " Iterating..." << std::endl;

        for (int iteration = 0; iteration < niter; iteration++) {
            if (verbose)
                std::cout << std::setw(3) << iteration + 1 << std::endl;

            project(flat_rec, flat_points, weights, npart, nprof);
            normalize(flat_rec, nprof, nbins);

            find_difference_profile(diff_prof, flat_rec, flat_profiles, all_bins);

            discr[iteration] = discrepancy(diff_prof, nprof, nbins);

            compensate_particle_amount(diff_prof, rparts, nprof, nbins);

            back_project(weights, flat_points, diff_prof, npart, nprof);
            clip(weights, npart, 0.0);

            if (sum(weights, npart) <= 0.)
                throw std::runtime_error("All of phase space got reduced to zeroes");

            callback(iteration + 1, niter);
        } //end for

        // Calculating final discrepancy
        project(flat_rec, flat_points, weights, npart, nprof);
        normalize(flat_rec, nprof, nbins);

        find_difference_profile(diff_prof, flat_rec, flat_profiles, all_bins);
        discr[niter] = discrepancy(diff_prof, nprof, nbins);

        callback(niter, niter);
    } catch (const std::exception &e) {
        cleanup();

        throw;
    }

    cleanup();

    if (verbose)
        std::cout << " Done!" << std::endl;
}
