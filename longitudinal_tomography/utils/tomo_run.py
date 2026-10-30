"""Module containing function for full Fortran
style tomographic reconstruction.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from ..data import data_treatment as dtreat
from ..tomography import tomography as tomography
from ..tracking import particles as pts
# Tomo modules
from ..tracking import tracking as tracking
from ..utils import tomo_data as tdata, tomo_output as tomoout
from ..utils import tomo_config as conf

from ..compat import tomoscope as tscp

if TYPE_CHECKING:
    from numpy.typing import NDArray as NPArray

log = logging.getLogger(__name__)


def run(input: str, reconstruct_profile: bool = None,
        output_dir: str = None, tomoscope: bool = False,
        plot: bool = False) \
        -> tuple[NPArray, NPArray, NPArray]:
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
    mode: Mode
        Decide which execution mode should be used

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

    tomo_data = tdata.TomographyData.from_file(input)

    if tomoscope:
        print(' Start')

    # Builds the machine, treats the raw data and fits synch_part_x if needed
    profiles = tomo_data.to_profiles()
    machine = profiles.machine

    if output_dir is None or output_dir == '':
        output_dir = machine.output_dir
    else:
        log.info(f'Overriding output dir with {output_dir}')

    # TODO: Insert space charge calculation from example file

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
    tomo = tomography.Tomography(profiles.waterfall, xp, yp)
    weight = tomo.run(niter=machine.niter, verbose=tomoscope)

    if tomoscope:
        for film in range(machine.filmstart, machine.filmstop + 1,
                          machine.filmstep):
            tscp.save_image(xp, yp, weight, machine.nbins, film, output_dir)
            tscp.save_profile(tomo.waterfall[film], film, output_dir)

        tscp.save_difference(tomo.diff, output_dir, film)

    # TODO mode is new and only phase_space is then in GPU. Discuss if necessary
    t_range, E_range, phase_space = dtreat.phase_space(tomo, machine, reconstr_idx)

    # Removing (if any) negative areas.
    phase_space = phase_space.clip(0.0)
    # Normalizing phase space.
    # See line 147
    phase_space /= conf.sum(phase_space)

    if plot:
        tomoout.show(phase_space, tomo.diff, profiles.waterfall[reconstr_idx])

    return t_range, E_range, phase_space
