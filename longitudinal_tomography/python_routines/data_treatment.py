"""Module containing operations for modifying the data derived from the cpp code.

:Author(s): **Bernardo Abreu Figueiredo**
"""

from ..utils import tomo_config as conf

def make_phase_space(xp: conf.ndarray[conf.int32],
                     yp: conf.ndarray[conf.int32],
                     weights: conf.ndarray,
                     n_bins: int, result_cpu: bool = True) -> conf.ndarray:
    index = yp + xp * n_bins

    phase_space = conf.zeros(n_bins**2)
    conf.add.at(phase_space, index, weights)

    phase_space = phase_space.reshape((n_bins, n_bins))
    if result_cpu:
        return phase_space.get()
    else:
        return phase_space