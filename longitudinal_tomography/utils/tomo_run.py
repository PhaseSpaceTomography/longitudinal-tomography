"""Module containing function for full Fortran
style tomographic reconstruction.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""

import logging
import typing as t
import numpy as np

from ..data import data_treatment as dtreat
from ..tomography import tomography as tomography
from ..tracking import particles as pts
# Tomo modules
from ..tracking import tracking as tracking
from ..utils import tomo_input as tomoin, tomo_output as tomoout

from ..compat import tomoscope as tscp

log = logging.getLogger(__name__)


def run(input: str, reconstruct_profile: bool = None,
        output_dir: str = None, tomoscope: bool = False,
        plot: bool = False) \
        -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Function to perform full reconstruction based on the original
    algorithm.

    Parameters
    ----------
    input: string
        Path to input file.
    reconstruct_profile: int, optional, default=None
        Profile to be reconstructed. If not provided, machine.filmstart
        will be reconstructed.
    output_dir: str
        Output directory when saving tomoscope output.
    tomoscope: bool
        Enable tomoscope specific output for progress tracking.
    plot: bool = False
        Plot phase space after reconstruction.

    Returns
    -------
    Phase axis: ndarray
        1D array containing time axis of reconstructed phase space image.
    Energy axis: ndarray
        1D array containing energy axis of reconstructed phase space image.
    density: ndarray
        2D array containing the reconstructed phase space image.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import longitudinal_tomography.utils.run_tomo as tomorun
    >>>
    >>> filepath = '...my/favourite/input.dat'
    >>> tRange, ERange, density = tomorun.run_file(filepath)
    >>>
    >>> vmin = np.min(density[density>0])
    >>> vmax = np.max(density)
    >>> plt.contourf(tRange*1E9, ERange/1E6, density.T,
                     levels=np.linspace(vmin, vmax, 50), cmap='Oranges')
    >>> plt.xlabel('dt (ns)')
    >>> plt.ylabel('dE (MeV)')
    >>> plt.show()
    """

    raw_params, raw_data = tomoin.get_user_input(input)

    if tomoscope:
        print(' Start')

    machine, frames = tomoin.txt_input_to_machine(raw_params)
    machine.values_at_turns()
    waterfall = frames.to_waterfall(raw_data)

    if output_dir is None or output_dir == '':
        output_dir = machine.output_dir
    else:
        log.info(f'Overriding output dir with {output_dir}')

    profiles = tomoin.raw_data_to_profiles(
        waterfall, machine, frames.rebin, frames.sampling_time)
    profiles.calc_profilecharge()

    # TODO: Insert space charge calculation from example file

    if profiles.machine.synch_part_x < 0:
        fit_info = dtreat.fit_synch_part_x(profiles)
        machine.load_fitted_synch_part_x_ftn(fit_info)

    if reconstruct_profile is None:
        reconstr_idx = machine.filmstart
    else:
        reconstr_idx = reconstruct_profile

    # Tracking...
    tracker = tracking.Tracking(machine)

    if tomoscope:
        tracker.enable_fortran_output(profiles.profile_charge)

    if tracker.self_field_flag:
        profiles.calc_self_fields()
        tracker.enable_self_fields(profiles)

    xp, yp = tracker.track(reconstr_idx)

    # Converting from physical coordinates ([rad], [eV])
    # to phase space coordinates.
    if not tracker.self_field_flag:
        xp, yp = pts.physical_to_coords(
            xp, yp, machine, tracker.particles.xorigin,
            tracker.particles.dEbin)

    # Filters out lost particles, transposes particle matrix,
    # casts to np.int32.
    xp, yp = pts.ready_for_tomography(xp, yp, machine.nbins)

    # Tomography!
    tomo = tomography.TomographyCpp(profiles.waterfall, xp, yp)
    weight = tomo.run(verbose=tomoscope)

    if tomoscope:
        for film in range(machine.filmstart, machine.filmstop + 1,
                          machine.filmstep):
            tscp.save_image(xp, yp, weight, machine.nbins, film, output_dir)
            tscp.save_profile(tomo.waterfall[film], film, output_dir)

        tscp.save_difference(tomo.diff, output_dir, film)

    t_range, E_range, phase_space = dtreat.phase_space(tomo, machine,
                                                       reconstr_idx)

    # Removing (if any) negative areas.
    phase_space = phase_space.clip(0.0)
    # Normalizing phase space.
    phase_space /= np.sum(phase_space)

    if plot:
        tomoout.show(phase_space, tomo.diff, profiles.waterfall[reconstr_idx])

    return t_range, E_range, phase_space
