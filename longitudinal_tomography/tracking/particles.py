"""
Module containing the Particles class for creating and storing
and initial particle distribution.

Module also contains utility functions for particle distribution
like assertions, conversions and filtering of lost particles.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
import logging
from typing import Tuple, Sequence, TYPE_CHECKING, Union

import numpy as np

from . import phase_space_info as psi
from .. import assertions as asrt, exceptions as expt

if TYPE_CHECKING:
    from .machine import Machine
    from .machine_base import MachineABC

log = logging.getLogger(__name__)


class Particles(object):
    """Class holding the initial particle distribution to be tracked.

    An automatic homogeneous particle based on the algorithm from the original
    tomography program can be created, by calling
    :func:`homogeneous_distribution`.

    Attributes
    ----------
    dEbin: float, default=None
        Energy size of bins in phase space coordinate system.
    xorigin: float, default=None
        The absolute difference (in bins) between phase=0 and the origin
        of the reconstructed phase space coordinate system.
    imin: int, default=None
        Minimum phase of reconstruction area. Given in bins phase space.
    imax: int, default=None
        Maximum phase of reconstruction area. Given in bins phase space.
    jmin: ndarray, default=None
        Minimum energy for each phase of the phase space coordinate system.
    jmax: ndarray, default=None
        Maximum energy for each phase of the phase space coordinate system.
    coordinates_dphi_denergy: tuple, default=(None, None)
        Tuple containing coordinates of initial particle distribution as 1D
        ndarrays (dphi, denergy). These are the differences in phase [rad]
        energy [eV] relative to the synchronous particle.
    """

    def __init__(self):
        self.dEbin = None
        self.xorigin = None

        # Area where particles have been automatically populated.
        # I - bin in phase axis, J - bin in energy axis
        self.imin = None
        self.imax = None
        self.jmin = None
        self.jmax = None

        self._dphi = None
        self._denergy = None

    @property
    def coordinates_dphi_denergy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Particle coordinates defined as @property.
        Use this property to provide and retrieve coordinates
        of the initial particle distribution in units of phase [rad]
        and energy [eV] relative to the synchronous particle.

        This property contains a built-in assertion of the provided particles.

        Returns
        -------
        coordinates: tuple
            Returns a tuple holding (dphi, denergy)

            * dphi
                ndarray containing the phase [rad] of each particle
                reative to the synchronous particle.
            * denergy
                ndarray containing the energy [eV] of each particle
                reative to the synchronous particle.
        """
        return self._dphi, self._denergy

    @coordinates_dphi_denergy.setter
    def coordinates_dphi_denergy(self,
                                 coordinates: Tuple[np.ndarray, np.ndarray]):
        """
        See the definition of the property
        :py:func`longitudinal_tomography.tracking.particles.coordinates_dphi_denergy` for full
        documentation.

        Parameters
        ----------
        coordinates: tuple
            Tuple holding (dphi, denergy)

            * dphi
                ndarray containing the phase [rad] of each particle
                relative to the synchronous particle.
            * denergy
                ndarray containing the energy [eV] of each particle
                relative to the synchronous particle.
        """
        self._dphi, self._denergy = _assert_coordinates(coordinates)

    def homogeneous_distribution(self, machine: 'Machine', recprof: int):
        """Function for automatic generation of particle distribution.

        The distributions created are identical to the distributions created
        in the Fortran tomography. The reconstruction area is found by calling
        :func:`~longitudinal_tomography.tracking.phase_space_info.PhaseSpaceInfo.find_binned_phase_energy_limits`.

        Parameters
        ----------
        machine: Machine
            Object containing the machine settings during the measurements.
            Needed for the calculation of the reconstruction area, and
            for setting the the number of particles to be created.
        recprof: int
            The index of the profile (time frame) to be reconstructed.
            This will be the profile where the distribution is generated.

        Raises
        ------
        MachineParameterError: Exception
            Raised if fields are missing from provided machine object.
        """
        self._assert_machine(machine)

        # The reconstruction area is found
        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.find_binned_phase_energy_limits()

        self.dEbin = psinfo.dEbin
        machine.dEbin = self.dEbin
        self.xorigin = psinfo.xorigin

        nbins_y = np.sum(psinfo.jmax[psinfo.imin: psinfo.imax + 1]
                         - psinfo.jmin[psinfo.imin: psinfo.imax + 1])

        # Creating the distribution of particles within one cell of the phase
        # space coordinate system.
        bin_pts = ((2.0 * np.arange(1, machine.snpt + 1) - 1)
                   / (2.0 * machine.snpt))

        # Creating x coordinates
        x = np.arange(psinfo.imin, psinfo.imax + 1, dtype=float)
        nbins_x = len(x)
        x = np.repeat(x, machine.snpt)
        x += np.tile(bin_pts, nbins_x)

        # Creating y coordinates.
        nbins_y = np.max(psinfo.jmax) - np.min(psinfo.jmin)
        y = np.arange(np.min(psinfo.jmin), np.max(psinfo.jmax), dtype=float)
        y = np.repeat(y, machine.snpt)
        y += np.tile(bin_pts, nbins_y)

        coords = np.meshgrid(x, y)
        coords = np.array([coords[0].flatten(), coords[1].flatten()])

        # Remove particles outside of the limits of i (phase) and j (energy)
        # of the reconstruction area.
        coords = coords[:, coords[1] < psinfo.jmax[coords[0].astype(int)]]
        coords = coords[:, coords[1] > psinfo.jmin[coords[0].astype(int)]]

        self.imin = psinfo.imin
        self.imax = psinfo.imax
        self.jmin = psinfo.jmin
        self.jmax = psinfo.jmax

        # Converting from phase space coordinates to physical units.
        coords = self._bin_nr_to_physical_coords(coords, machine, recprof)
        self.coordinates_dphi_denergy = coords

    def _bin_nr_to_physical_coords(self,
                                   coordinates: Union[
                                       Sequence[np.ndarray],
                                       np.ndarray
                                   ],
                                   machine: 'Machine', recprof: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Function to convert from reconstructed phase space coordinates
        to physical units.

        Needed by the homogeneous_distribution routine.

        Parameters
        ----------
        coordinates: tuple (ndarray:float, ndarray:float)
            Coordinates of particles (I coordinate, J coordinate)
            provided as fractions of bins of the reconstructed phase space
            coordinate system.
        machine: Machine
            Machine object holding machine parameters and settings
        recprof: int
            Profile to reconstruct.
        """
        turn = recprof * machine.dturns
        dphi = ((coordinates[0] + self.xorigin)
                * machine.h_num * machine.omega_rev0[turn] * machine.dtbin
                - machine.phi0[turn])
        denergy = (coordinates[1] - machine.synch_part_y) * self.dEbin
        return dphi, denergy

    # Assertions to assure that all needed fields are provided in the
    # given machine object
    def _assert_machine(self, machine: 'Machine'):
        needed_fieds = ['snpt', 'h_num', 'omega_rev0', 'eta0',
                        'dtbin', 'phi0', 'synch_part_y', 'dturns', 'phi12',
                        'nbins', 'beam_ref_frame', 'full_pp_flag',
                        'demax', 'e0', 'min_dt', 'max_dt', 'time_at_turn']
        asrt.assert_fields(
            machine, 'machine', needed_fieds, expt.MachineParameterError,
            'Did you remember to use machine.values_at_turns()?')


def filter_lost(xp: np.ndarray, yp: np.ndarray, img_width: int) \
        -> Tuple[np.ndarray, np.ndarray, int]:
    """Remove lost particles (particles that leaves the image width).

    Parameters
    ----------
    xp: ndarray
        2D array containing x-coordinates of all particles at each time frame.
        Particle coordinates must be given in phase space coordinates
        as integers. Shape: (nprofiles, nparts).
    yp: ndarray, int
        2D array containing y-coordinates of all particles at each time frame.
        Particle coordinates must be given in phase space coordinates
        as integers. Shape: (nprofiles, nparts).
    img_width: int
        Number of bins in measured profiles.

    Returns
    -------
    xp: ndarray
        2D array containing x-coordinates of all particles at each time frame.
        Now containing only particles inside of the image width.
        Shape: (nprofiles, nparts).
    yp: ndarray
        2D array containing y-coordinates of all particles at each time frame.
        Now containing only particles inside of the image width.
        Shape: (nprofiles, nparts).
    nr_lost_points: int
        Number of particles removed.

    """
    nr_lost_pts = 0

    # Find all particles outside of image width
    invalid_pts = np.argwhere(np.logical_or(xp >= img_width, xp < 0))

    if np.size(invalid_pts) > 0:
        # Mark particle as invalid only once
        invalid_pts = np.unique(invalid_pts.T[1])
        # Save number of invalid particles
        nr_lost_pts = len(invalid_pts)
        # Remove invalid particles
        xp = np.delete(xp, invalid_pts, axis=1)
        yp = np.delete(yp, invalid_pts, axis=1)

    if xp.size == 0:
        raise expt.InvalidParticleError(
            'All particles removed during filtering')

    return xp, yp, nr_lost_pts


def physical_to_coords(tracked_dphi: np.ndarray, tracked_denergy: np.ndarray,
                       machine: 'MachineABC', xorigin: float, dEbin: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Function to convert from physical units ([rad], [eV]) to reconstructed
    phase space coordinates (bin numbers).

    The phase space coordinates are needed for the tomographic reconstruction
    routines.

    Parameters
    ----------
    tracked_dphi: ndarray
        2D array containing the phase difference [rad] relative to
        the synchronous particle for every particle at every time frame.
        Array shape: (nprofiles, nparticles)
    tracked_denergy: ndarray
        2D array containing the energy difference [eV] relative to
        the synchronous particle for every particle at every time frame.
        Array shape: (nprofiles, nparticles)
    machine: MachineABC
        machine object holding machine parameters, and settings for
        the reconstruction.
    xorigin: float
        The absolute difference (in bins) between phase=0 and the origin
        of the reconstructed phase space coordinate system.
    dEbin: float
        Energy size of bins in phase space coordinate system.

    Returns
    -------
    xp: ndarray
        2D array containing the x-coordinate for every particle
        at every time frame, given fractions of bins of
        the phase space coordinate system.
        Array shape: (nprofiles, nparticles)
    yp: ndarray
        2D array containing the y-coordinate for every particle
        at every time frame, given fractions of bins of
        the phase space coordinate system.
        Array shape: (nprofiles, nparticles)
    """
    if tracked_dphi.shape != tracked_denergy.shape:
        raise expt.InvalidParticleError(
            'Different shape of arrays containing phase and energies')
    nprof = tracked_denergy.shape[0]

    profiles = np.arange(nprof)
    turns = profiles * machine.dturns

    xp = ((tracked_dphi
           + machine.phi0[turns].reshape(-1, 1))
          / (float(machine.h_num)
             * machine.omega_rev0[turns].reshape(-1, 1)
             * machine.dtbin) - xorigin)

    yp = (tracked_denergy
          / float(dEbin) + machine.synch_part_y)
    return xp, yp


def ready_for_tomography(xp: np.ndarray, yp: np.ndarray, nbins: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Function to prepare tracked particles tomography routine.

    Handy if particles are tracked using the functions of the
    :mod:`longitudinal_tomography.tracking.tracking` module.

    **Function does the following actions:**

    * Removes particles leaving the image width.
    * Transposes the coordinate arrays from (nprofiles, nparts)\
    to (nparts, nprofiles).
    * Casts coordinates to integers.

    The returned coordinates are ready to be used in the tomographic
    reconstruction routines.

    Parameters
    ----------
    xp: ndarray
        2D array containing the x-coordinate for every particle
        at every time frame, given fractions of bins of
        the phase space coordinate system.
        Array shape: **(N, M)**, where N is the number of profiles,
        and M is the number of particles.
    yp: ndarray, float
        2D array containing the y-coordinate for every particle
        at every time frame, given fractions of bins of
        the phase space coordinate system.
        Array shape: **(N, M)**, where N is the number of profiles,
        and M is the number of particles.

    Returns
    -------
    xp: ndarray
        2D array containing the x-coordinate for every particle
        at every time frame, given in phase space coordinates.
        Array shape: **(M, N)**, where N is the number of profiles,
        and M is the number of particles.
    yp: ndarray
        2D array containing the y-coordinate for every particle
        at every time frame, given in phase space coordinates.
        Array shape: **(M, N)**, where N is the number of profiles,
        and M is the number of particles.
    """
    xp, yp, lost = filter_lost(xp, yp, nbins)
    log.info(f'number of lost particles: {lost}')
    xp = xp.astype(np.int32).T
    yp = yp.astype(np.int32).T

    return xp, yp


# Function to check that coordinates are valid
def _assert_coordinates(coordinates: Sequence[np.ndarray]) \
        -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(coordinates, '__getitem__'):
        raise expt.InvalidParticleError('coordinates should be a sequence')
    if not len(coordinates) == 2:
        raise expt.InvalidParticleError('Two arrays of coordinates should be'
                                        'provided')
    for coord in coordinates:
        if not hasattr(coord, '__iter__'):
            raise expt.InvalidParticleError('coordinates should be iterable')
    if not len(coordinates[0]) == len(coordinates[1]):
        raise expt.InvalidParticleError(
            'arrays holding coordinates of x and y axis '
            'should have the same length')
    return coordinates[0], coordinates[1]
