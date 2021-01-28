#include "data_treatment.h"
#include <stdexcept>


double *make_phase_space(const int *const xp, const int *const yp, const double *const weight, const int n_particles, const int n_bins) {
    double * phase_space = new double[n_bins * n_bins]();
    int n_bins2 = n_bins * n_bins;

    for (int i = 0; i < n_particles; i++) {
        int index = yp[i] + xp[i] * n_bins;
        if (index >= n_bins2)
            throw std::runtime_error("Index out of bounds");
        phase_space[index] += weight[i];
    }

    return phase_space;
}
