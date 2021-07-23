"""
Macro functions provided for convenience

:Author(s): **Anton Lu**
"""
import typing as t
import numpy as np

from .tracking import Tracking
from .tracking.machine_base import MachineABC
from .tracking import particles
from .tomography import tomography


__all__ = ['track', 'tomogram']


def track(machine: MachineABC, reconstruction_idx: int = None,
          callback: t.Callable = None) \
        -> t.Tuple[np.ndarray, np.ndarray]:

    tracker = Tracking(machine)
    dphi, denergy = tracker.track(reconstruction_idx, callback=callback)
    xp, yp = particles.physical_to_coords(dphi, denergy, machine,
                                          tracker.particles.xorigin,
                                          tracker.particles.dEbin)
    xp, yp = particles.ready_for_tomography(xp, yp, machine.nbins)

    return xp, yp


def tomogram(waterfall: np.ndarray, xp: np.ndarray, yp: np.ndarray,
             n_iter: int, callback: t.Callable = None) \
        -> tomography.Tomography:

    tomo = tomography.Tomography(waterfall, xp, yp)
    tomo.run(n_iter, callback=callback)

    return tomo
