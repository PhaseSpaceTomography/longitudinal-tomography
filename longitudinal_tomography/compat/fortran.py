"""
Compatibility module. Everything in this submodule can be removed at once with
minimal impact to the rest of the package (some if-statements need to be
removed in the rest of the package as well).

The intention of this package is to provide Fortran I/O compatibility with the
tomoscope until it is deprecated.

:Author(s): **Anton Lu**
"""

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from .. import assertions as asrt, exceptions as expt

if TYPE_CHECKING:
    from ..tracking.machine import Machine
    from ..tracking.particles import Particles

log = logging.getLogger(__name__)


# Input-output
# --------------------------------------------------------------- #
#                           PROFILES                              #
# --------------------------------------------------------------- #

def save_profile(profiles: np.ndarray, recprof: int, output_dir: str):
    """Write phase-space image to text-file in the original format.
    The name of the file will be profileXXX.data, where XXX is the index
    of the time frame to be reconstructed counting from one.

    Parameters
    ----------
    profiles: ndarray
        Profile measurements as a 2D array with the shape: (nprofiles, nbins).
    recprof: int
        Index of profile to be saved.
    output_dir: string
        Path to output directory.
    """
    out_profile = profiles[recprof].flatten()
    file_path = os.path.join(output_dir, f'profile{recprof + 1:03d}.data')
    with open(file_path, 'w') as f:
        for element in out_profile:
            f.write(f' {element:0.7E}\n')


def save_self_volt_profile(self_fields: np.ndarray, output_dir: str):
    """Write self volts to text file in tomoscope format.

    Parameters
    ----------
    self_fields: ndarray
        Calculated self-field voltages.
    output_dir: string
        Path to output directory.
    """
    out_profile = self_fields.flatten()
    file_path = os.path.join(output_dir, 'vself.data')
    with open(file_path, 'w') as f:
        for element in out_profile:
            f.write(f' {element:0.7E}\n')


# --------------------------------------------------------------- #
#                         PHASE-SPACE                             #
# --------------------------------------------------------------- #

def save_phase_space(image: np.ndarray, recprof: int, output_path: str):
    """Save phase-space image in a tomoscope format.

    Parameters
    ----------
    image: ndarray
        2D array holding the weight of each cell of the recreated phase-space.
    recprof: int
        Index of reconstructed profile.
    output_path: string
        Path to the output directory.
    """
    log.info(f'Saving image{recprof} to {output_path}')
    image = image.flatten()
    file_path = os.path.join(output_path, f'image{recprof + 1:03d}.data')
    with open(file_path, 'w') as f:
        for element in image:
            f.write(f'  {element:0.7E}\n')


# --------------------------------------------------------------- #
#                           PROFILES                              #
# --------------------------------------------------------------- #


# --------------------------------------------------------------- #
#                          PLOT INFO                              #
# --------------------------------------------------------------- #

def write_plotinfo(machine: 'Machine', particles: 'Particles',
                   profile_charge: float) -> str:
    """Creates string of plot info needed for the original output
    for the tomography program.

    Parameters
    ----------
    machine: Machine
        Object containing machine parameters.
    particles: Particles
        Object containing particle distribution and information about
        the phase space reconstruction.
    profile_charge: float
        Total charge of a reference profile.

    Returns
    -------
    plot_info: string
        String containing information needed by the tomoscope application.
        The returned string has the same format as in the original version.

    """
    recprof = machine.filmstart
    rec_turn = recprof * machine.dturns

    # Check if a Fortran styled fit has been performed.
    fit_performed = True
    fit_info_vars = ['fitted_synch_part_x', 'bunchlimit_low', 'bunchlimit_up']
    for var in fit_info_vars:
        if not hasattr(machine, var) or getattr(machine, var) is None:
            fit_performed = False
            break

    if fit_performed:
        bunchlimit_low = machine.bunchlimit_low
        bunchlimit_up = machine.bunchlimit_up
        fitted_synch_part_x = machine.fitted_synch_part_x
    else:
        bunchlimit_low = 0.0
        bunchlimit_up = 0.0
        fitted_synch_part_x = 0.0

    if particles.dEbin is None:
        raise expt.EnergyBinningError(
            'dEbin has not been calculated for this '
            'phase space info object.\n'
            'Cannot print plot info.')
    if particles.imin is None or particles.imax is None:
        raise expt.PhaseLimitsError(
            'The limits in phase (I) has not been found '
            'for this phase space info object.\n'
            'Cannot print plot info.')

        # '+ 1': Converting from Python to Fortran indexing
    out_s = f' plotinfo.data\n' \
            f'Number of profiles used in each reconstruction,\n' \
            f' profilecount = {machine.nprofiles}\n' \
            f'Width (in pixels) of each image = ' \
            f'length (in bins) of each profile,\n' \
            f' profilelength = {machine.nbins}\n' \
            f'Width (in s) of each pixel = width of each profile bin,\n' \
            f' dtbin = {machine.dtbin:0.4E}\n' \
            f'Height (in eV) of each pixel,\n' \
            f' dEbin = {particles.dEbin:0.4E}\n' \
            f'Number of elementary charges in each image,\n' \
            f' eperimage = ' \
            f'{profile_charge:0.3E}\n' \
            f'Position (in pixels) of the reference synchronous point:\n' \
            f' xat0 =  {machine.synch_part_x:.3f}\n' \
            f' yat0 =  {machine.synch_part_y:.3f}\n' \
            f'Foot tangent fit results (in bins):\n' \
            f' tangentfootl =    {bunchlimit_low:.3f}\n' \
            f' tangentfootu =    {bunchlimit_up:.3f}\n' \
            f' fit xat0 =   {fitted_synch_part_x:.3f}\n' \
            f'Synchronous phase (in radians):\n' \
            f' phi0( {recprof + 1}) = {machine.phi0[rec_turn]:.4f}\n' \
            f'Horizontal range (in pixels) of the region in ' \
            f'phase space of map elements:\n' \
            f' imin( {recprof + 1}) =   {particles.imin} and ' \
            f'imax( {recprof + 1}) =  {particles.imax}'
    return out_s


# --------------------------------------------------------------- #
#                         DISCREPANCY                             #
# --------------------------------------------------------------- #

def save_difference(diff: np.ndarray, output_path: str, recprof: int):
    """Write reconstruction discrepancy to text file with original format.

    Parameters
    ----------
    diff: ndarray
        1D array containing the discrepancy for the phase space at each
        iteration of the reconstruction.
    output_path: string
        Path to output directory.
    recprof: int
        Index of profile to be saved.
    """
    log.info(f'Saving saving difference to {output_path}')
    file_path = os.path.join(output_path, f'd{recprof + 1:03d}.data')
    with open(file_path, 'w') as f:
        for i, d in enumerate(diff):
            f.write(f'         {i:3d}  {d:0.7E}\n')


# --------------------------------------------------------------- #
#                          TRACKING                               #
# --------------------------------------------------------------- #

def print_tracking_status(ref_prof: int, to_profile: int):
    """Write output for particle tracking in the original format.
    Since the original algorithm is somewhat different,
    the **output concerning lost particles is not valid**.
    Meanwhile, the format it is needed by the tomoscope application.
    Profile numbers are added by one in order to compensate for
    differences in Python and Fortran indexing. Fortran counts from
    one, Python counts from 0.
    This function is used in the tracking algorithm.

    Parameters
    ----------
    ref_prof: int
        Index of profile to be reconstructed.
    to_profile: int
        Profile to which the algorithm is currently tracking towards.
    """
    print(f' Tracking from time slice  {ref_prof + 1} to  '
          f'{to_profile + 1},   0.000% went outside the image width.')


# Data treatment

def calc_baseline(waterfall: np.ndarray, ref_prof: int,
                  percent: float = 0.05) -> float:
    """Function for finding baseline of raw data.

    The function is based on the original Fortran program,
    and uses a percentage of a reference profile in order to
    find the baseline of the measurements.

    Parameters
    ----------
    waterfall: ndarray
        Raw-data shaped as waterfall: (nprofiles, nbins).
    ref_prof: int
        Index of reference profile.
    percent: float, optional, default=0.05
        A number between 0 and 1 describing the percentage of the
        reference profile used to find the baseline.

    Returns
    -------
    baseline: float
        Baseline of reference profile.

    Raises
    ------
    InputError: Exception
        Raised if percentage is not given as a float between 0 and 1.

    """
    asrt.assert_inrange(percent, 'percent', 0.0, 1.0, expt.InputError,
                        'The chosen percent of raw_data '
                        'to create baseline from is not valid')

    nbins = len(waterfall[ref_prof])
    iend = int(percent * nbins)

    return np.sum(waterfall[ref_prof, :iend]) / np.floor(percent * nbins)
