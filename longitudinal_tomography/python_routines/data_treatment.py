"""Module containing operations for modifying the data derived from the cpp code.

:Author(s): **Bernardo Abreu Figueiredo**
"""

import numpy as np
from ..utils.execution_mode import Mode

def make_phase_space(xp: np.ndarray[np.int32],
                     yp: np.ndarray[np.int32],
                     weights: np.ndarray[np.float64],
                     n_bins: int, mode: Mode = Mode.JIT) -> np.ndarray[np.float64]:
    index = yp + xp * n_bins

    if(mode == Mode.CUPY or mode == Mode.CUDA):
        import cupy as cp
        phase_space = cp.empty(n_bins**2)
        cp.add.at(phase_space, index, weights)
    else:
        phase_space = np.empty(n_bins**2)
        np.add.at(phase_space, index, weights)

    return phase_space.reshape((n_bins, n_bins))