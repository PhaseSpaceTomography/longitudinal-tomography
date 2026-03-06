/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file data_treatment.h
 *
 * C++ equivalent of the `tomo.data.data_treatment` module.
 */

#ifndef TOMOGRAPHY_DATA_TREATMENT_H
#define TOMOGRAPHY_DATA_TREATMENT_H

template <typename real_int_t, typename int_t>
real_int_t *make_phase_space(const int_t *const xp,
                             const int_t *const yp,
                             const real_int_t *const weight,
                             const int n_particles,
                             const int n_bins);

#endif