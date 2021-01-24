from typing import Tuple
import numpy as np

from .tracking import Tracking
from .tracking import Machine
from .tracking import particles


def track(machine: Machine, reconstruction_idx: int = None) \
        -> Tuple[np.ndarray, np.ndarray]:

    tracker = Tracking(machine)
    dphi, denergy = tracker.track(reconstruction_idx)
    xp, yp = particles.physical_to_coords(dphi, denergy, machine,
                                          tracker.particles.xorigin,
                                          tracker.particles.dEbin)
    xp, yp = particles.ready_for_tomography(xp, yp, machine.nbins)

    return xp, yp
