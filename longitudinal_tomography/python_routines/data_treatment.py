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