"""Module containing functions for handling output from tomography programs.

Every function ending on 'ftn' creates an
output equal original Fortran program.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from ..cpp_routines import libtomo
from ..compat import fortran


def save_profile_ftn(*args, **kwargs):
    """Here for backwards compatibility. See
    :func:`~longitudinal_tomography.compat.fortran.save_profile`"""
    # warn('This function has been moved to '
    #      'longitudinal_tomography.compat.fortran')
    return fortran.save_profile(*args, **kwargs)


def save_self_volt_profile_ftn(*args, **kwargs):
    """Here for backwards compatibility. See
    :func:`~longitudinal_tomography.compat.fortran.save_self_volt_profile`"""
    # warn('This function has been moved to '
    #      'longitudinal_tomography.compat.fortran')
    return fortran.save_self_volt_profile(*args, **kwargs)


def save_phase_space_ftn(*args, **kwargs):
    """Here for backwards compatibility. See
    :func:`~longitudinal_tomography.compat.fortran.save_phase_space`"""
    # warn('This function has been moved to '
    #      'longitudinal_tomography.compat.fortran')
    return fortran.save_phase_space(*args, **kwargs)


def write_plotinfo_ftn(*args, **kwargs):
    """Here for backwards compatibility. See
    :func:`~longitudinal_tomography.compat.fortran.write_plotinfo`"""
    # warn('This function has been moved to '
    #      'longitudinal_tomography.compat.fortran')
    return fortran.write_plotinfo(*args, **kwargs)


def save_difference_ftn(*args, **kwargs):
    """Here for backwards compatibility. See
    :func:`~longitudinal_tomography.compat.fortran.save_difference`"""
    # warn('This function has been moved to '
    #      'longitudinal_tomography.compat.fortran')
    return fortran.save_difference(*args, **kwargs)


def print_tracking_status_ftn(*args, **kwargs):
    """Here for backwards compatibility. See
    :func:`~longitudinal_tomography.compat.fortran.print_tracking_status`"""
    # warn('This function has been moved to '
    #      'longitudinal_tomography.compat.fortran')
    return fortran.print_tracking_status(*args, **kwargs)


def create_phase_space_image(
        xp: np.ndarray, yp: np.ndarray, weight: np.ndarray, n_bins: int,
        recprof: int) -> np.ndarray:
    """Convert from weighted particles to phase-space image.

    The output is equal to the phase space image created
    in the original version.

    Parameters
    ----------
    xp: ndarray
        2D array containing the x coordinates of every
        particle at every time frame. Must be given in coordinates
        of the phase space coordinate system as integers.
        Shape: (N, M), where N is the number of particles and
        M is the number of profiles.
    yp: ndarray
        2D array containing the y coordinates of every
        particle at every time frame. Must be given in coordinates
        of the phase space coordinate system as integers.
        Shape: (N, M), where N is the number of particles and
        M is the number of profiles.
    weight: ndarray
        1D array containing the weight of each particle.
    n_bins: int
        Number of bins in a profile measurement.
    recprof: int
        Index of reconstructed profile.

    Returns
    -------
    phase_space: ndarray
        Phase space presented as 2D array with shape (N, N),
        where N is the number of bins in a profile. This
        phase space image has the same format as from the original program.
    """

    phase_space = libtomo.make_phase_space(xp[:, recprof].astype(np.int32),
                                           yp[:, recprof].astype(np.int32),
                                           weight, n_bins)

    # Removing (if any) negative areas.
    phase_space = phase_space.clip(0.0)
    # Normalizing phase space.
    phase_space /= np.sum(phase_space)
    return phase_space


# --------------------------------------------------------------- #
#                         END PRODUCT                             #
# --------------------------------------------------------------- #

def show(image: np.ndarray, diff: np.ndarray, rec_prof: np.ndarray,
         figure: plt.figure = None):
    """Nice presentation of reconstruction.

    Parameters
    ----------
    image: ndarray
        Recreated phase-space image.
        Shape: (N, N), where N is the number of profile bins.
    diff: ndarray
        1D array containing discrepancies for each iteration of reconstruction.
    rec_prof: ndarray
        1D array containing the measured profile to be reconstructed.
    """

    # Normalizing recprof:
    rec_prof[:] /= np.sum(rec_prof)

    # Creating plot
    gs = gridspec.GridSpec(4, 4)

    if figure is not None:
        fig = figure
    else:
        fig = plt.figure()

    img = fig.add_subplot(gs[1:, :3])
    profs1 = fig.add_subplot(gs[0, :3])
    profs2 = fig.add_subplot(gs[1:4, 3])
    convg = fig.add_subplot(gs[0, 3])

    cimg = img.imshow(image.T, origin='lower',
                      interpolation='nearest', cmap='hot')

    profs1.plot(np.sum(image, axis=1), label='reconstructed', zorder=5)
    profs1.plot(rec_prof, label='measured', zorder=0)
    profs1.legend()

    profs2.plot(np.sum(image, axis=0),
                np.arange(image.shape[0]))

    convg.plot(diff, label='discrepancy')
    convg.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    convg.legend()

    for ax in (profs1, profs2, convg):
        ax.set_xticks([])
        ax.set_yticks([])

    convg.set_xticks(np.arange(len(diff)))
    convg.set_xticklabels([])

    if figure is None:
        fig.set_size_inches(8, 8)
        fig.tight_layout()

        plt.show()
