#include "data_treatment.h"


double *make_phase_space(const int *const xp, const int *const yp, const double *const weight, const int n_particles, const int n_bins) {
    double * phase_space = new double[n_bins * n_bins]();

#pragma omp parallel for
    for (int i = 0; i < n_particles; i++)
        phase_space[yp[i] + xp[i] * n_bins] += weight[i];

    return phase_space;
}
