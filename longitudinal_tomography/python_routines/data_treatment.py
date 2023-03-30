"""Module containing operations for modifying the data derived from the cpp code.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np

def make_phase_space(xp: np.ndarray[np.int32],
                     yp: np.ndarray[np.int32],
                     weights: np.ndarray[np.float64],
                     n_particles: int, # TODO: where do I get that from?
                     n_bins: int) -> np.ndarray[np.float64]:
    phase_space = np.empty(n_bins**2)
    
    for i in range(n_particles):
        index = yp[i] + xp[i] * n_bins
        if index >= n_bins**2:
            raise Exception("Index out of bounds")
            exit()
        phase_space[index] += weights[i]


#double *make_phase_space(const int *const xp, const int *const yp, const double *const weight, const int n_particles,
#                         const int n_bins) {
#    double *phase_space = new double[n_bins * n_bins]();
#    int n_bins2 = n_bins * n_bins;
#
#    for (int i = 0; i < n_particles; i++) {
#        int index = yp[i] + xp[i] * n_bins;
#        if (index >= n_bins2)
#            throw std::out_of_range("Index out of bounds");
#        phase_space[index] += weight[i];
#    }
#
#    return phase_space;
#}