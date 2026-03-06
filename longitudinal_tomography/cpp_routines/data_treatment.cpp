/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file data_treatment.cpp
 *
 * Functions relating to the `tomo.data.data_treatment` module of the tomography.
 */

#include <stdexcept>
#include "data_treatment.h"


/**
 *
 * @param xp A `n_particles` long array with bin numbers on the x axis
 * @param yp A `n_particles` long array with bin numbers on the y axis
 * @param weight A `n_particles` long array with the weights of each particle
 * @param n_particles Number of particles
 * @param n_bins Number of bins in the phase space
 * @return A n_bins*n_bins array (1d) representing the density plot of the phase space
 */
template <typename real_int_t, typename int_t>
real_int_t *make_phase_space(const int_t *const xp,
                             const int_t *const yp,
                             const real_int_t *const weight,
                             const int n_particles,
                             const int n_bins) {
    real_int_t *phase_space = new real_int_t[n_bins * n_bins]();
    int n_bins2 = n_bins * n_bins;

    for (int i = 0; i < n_particles; i++) {
        int index = yp[i] + xp[i] * n_bins;
        if (index >= n_bins2)
            throw std::out_of_range("Index out of bounds");
        phase_space[index] += weight[i];
    }

    return phase_space;
}

template double *make_phase_space(const int32_t *const xp,
                                  const int32_t *const yp,
                                  const double *const weight,
                                  const int n_particles,
                                  const int n_bins);

template float *make_phase_space(const int32_t *const xp,
                                 const int32_t *const yp,
                                 const float *const weight,
                                 const int n_particles,
                                 const int n_bins);

template double *make_phase_space(const int64_t *const xp,
                                  const int64_t *const yp,
                                  const double *const weight,
                                  const int n_particles,
                                  const int n_bins);

template float *make_phase_space(const int64_t *const xp,
                                 const int64_t *const yp,
                                 const float *const weight,
                                 const int n_particles,
                                 const int n_bins);

template int32_t *make_phase_space(const int32_t *const xp,
                                   const int32_t *const yp,
                                   const int32_t *const weight,
                                   const int n_particles,
                                   const int n_bins);

template int64_t *make_phase_space(const int64_t *const xp,
                                   const int64_t *const yp,
                                   const int64_t *const weight,
                                   const int n_particles,
                                   const int n_bins);
