"""
Macro functions provided for convenience

:Author(s): **Anton Lu**
"""
from __future__ import annotations
import typing as t
import numpy as np

from .tracking import Tracking
from .tracking.machine_base import MachineABC
from .tracking import particles
from .tomography import tomography
from .utils import tomo_input as tin
from .data import data_treatment as dtreat


__all__ = ['track', 'tomogram']


def read_input_file(input_file_path: str) -> t.Tuple[MachineABC, np.ndarray]:
    """
    Read input file and return a machine and a waterfall

    :param input_file_path: path to input file
    :return: machine and waterfall
    """
    raw_params, raw_data = tin.get_user_input(input_file_path)

    machine, frames = tin.txt_input_to_machine(raw_params)
    machine.values_at_turns()
    measured_waterfall = frames.to_waterfall(raw_data)

    profiles = tin.raw_data_to_profiles(
        measured_waterfall, machine, frames.rebin, frames.sampling_time)
    profiles.calc_profilecharge()

    if profiles.machine.synch_part_x < 0:
        fit_info = dtreat.fit_synch_part_x(profiles)
        machine.load_fitted_synch_part_x_ftn(fit_info)

    return machine, profiles.waterfall


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
