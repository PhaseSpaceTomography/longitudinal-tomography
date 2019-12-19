import numpy as np
import logging as log

from . import phase_space_info as psi
from ..utils import assertions as asrt
from ..utils import exceptions as expt


# This class sets up the inital particle distribution of the test particles.
# The class is NOT NEEDED for the tracking of the particles, but is meant as
#  a utility.
#
# By using the 'homogeneous_distribution' function, a fortran styled
# distrubution is created. Here, each cell of the phase-space
# is populated by snpt^2 particles. Snpt is the square root of the
# number of test particles pr cell. This is given by the input parameters.
# The particles will be saved in coordinates given as fractions of bins.
# The bins are determined by the machine parameters, and the waterfall
# (measured profiles). By adding a Projection (timespace) object,
# the necessary values will be calcualted.
#
# One can also give the particles initial positions manually.
# In this case, the particles will be given to the object using the
# 'set_coordinates' function. The input to this function should
# be coordinates in fractions of bins, together with the relevant
# projection (timespace) object.
# 
# In order to use the coordinates in the particle tracker, the
# coordinates must be converted from the binned coordinate system
# to physical units. This is done using the 'init_coords_to_physical'
# function. This will convert the one-dimensional array containing
# the initial coordinates to values in phase and energy.
# 
# You can evade the usage of this object for initializing the particles.
# This can be done by directly providing the particle tracker object
# with arrays containing the coodinates in units of phase and energy.
#
# In order to use the tracked particles further in the program,
# they must be mapped to the binned coordinate system. This
# can be done by using the static 'physical_to_coords' function.
# Here, a two-dimensional array containing the output
# of the tracking routine mapped to the binned coodinate system.

class Particles(object):

    # x- and y-coords are the coordinates of each particle, given in
    # fractions of bins. 
    def __init__(self):       
        self.dEbin = None
        self.xorigin = None

        # Area where particles have been automaticaly populated.
        # I - bin in phase axis, J - bin in energy axis 
        self.imin = None
        self.imax = None
        self.jmin = None
        self.jmax = None

        self._dphi = None
        self._denergy = None

    @property
    def coordinates_dphi_denergy(self):
        return (self._dphi, self._denergy)

    @coordinates_dphi_denergy.setter
    def coordinates_dphi_denergy(self, coordinates):
        self._dphi, self._denergy = _assert_coordinates(coordinates)

    # The function wil create a homogeneous distribution of particles within
    # an area defined by the user. The area is given by the i and jlimits
    # found in the _psinfo object. Depending on the 'full_pp_flag' set in
    # the input parameters, the particles will be distribted in the bucket
    # area or the full image width.
    # This creates a particle distribution resembeling the original Fortran
    # version.
    # The particles coordinates will be saved as fractions of bins in
    # the x (phase) and y (energy) axis.
    def homogeneous_distribution(self, machine, recprof):
        self._assert_machine(machine)
        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.find_binned_phase_energy_limits()
        self.dEbin = psinfo.dEbin
        self.xorigin = psinfo.xorigin

        nbins_y = np.sum(psinfo.jmax[psinfo.imin: psinfo.imax + 1]
                        - psinfo.jmin[psinfo.imin: psinfo.imax + 1])

        # creating the distribution of particles within one cell.
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

        # Remove particles outside of the ijlimits.
        coords = coords[:,coords[1] < psinfo.jmax[coords[0].astype(int)]]
        coords = coords[:, coords[1] > psinfo.jmin[coords[0].astype(int)]]

        self.imin = psinfo.imin
        self.imax = psinfo.imax
        self.jmin = psinfo.jmin
        self.jmax = psinfo.jmax

        coords = self._bin_nr_to_physical_coords(coords, machine, recprof)
        self.coordinates_dphi_denergy = coords

    # Convert particle coordinates from coordinates as fractions of bins,
    # to physical units. The physical units are phase (x-axis),
    # and energy (y-axis).
    # This format is needed for the particle tracking routine.  
    def _bin_nr_to_physical_coords(self, coordinates, machine, recprof):
        turn = recprof * machine.dturns
        dphi = ((coordinates[0] + self.xorigin)
                * machine.h_num * machine.omega_rev0[turn] * machine.dtbin
                - machine.phi0[turn])
        denergy = (coordinates[1] - machine.synch_part_y) * self.dEbin
        return dphi, denergy

    def _assert_machine(self, machine):
        needed_fieds = ['snpt', 'h_num', 'omega_rev0', 'eta0',
                        'dtbin', 'phi0', 'synch_part_y', 'dturns', 'phi12',
                        'nbins', 'beam_ref_frame', 'full_pp_flag',
                        'demax', 'vrf2', 'vrf2dot', 'e0', 'vrf1',
                        'vrf1dot', 'min_dt', 'max_dt', 'time_at_turn']
        asrt.assert_fields(
            machine, 'machine', needed_fieds, expt.MachineParameterError,
            'Did you remember to use machine.values_at_turns()?')

# Takes particles in phase space coordinates.
# Remove particles which left image width.
# Returns filtered x- and y-coordinates,
# and number of particles that were lost
def filter_lost(xp, yp, img_width):
    nr_lost_pts = 0

    # Find all invalid particle values
    invalid_pts = np.argwhere(np.logical_or(xp >= img_width, xp < 0))
            
    if np.size(invalid_pts) > 0:
        # Find all invalid particles
        invalid_pts = np.unique(invalid_pts.T[1])
        nr_lost_pts = len(invalid_pts)
        # Removing invalid particles
        xp = np.delete(xp, invalid_pts, axis=1)
        yp = np.delete(yp, invalid_pts, axis=1)

    return xp, yp, nr_lost_pts


# Convert from physical units to coordinates in bins.
# Input is a two-dimensional array containing the tracked points.
# The output is a two-dimensional aray containing the tracked
# points given as nr of bins in phase and energy.
def physical_to_coords(tracked_dphi, tracked_denergy,
                       machine, xorigin, dEbin):
    if tracked_dphi.shape != tracked_denergy.shape:
        raise AssertionError('Different shape of arrays containing '
                             'phase and energies')
    nprof = tracked_denergy.shape[0]

    profiles = np.arange(nprof)
    turns = profiles * machine.dturns

    xp = np.zeros(tracked_dphi.shape)
    yp = np.zeros(tracked_dphi.shape)

    xp[profiles] = ((tracked_dphi[profiles] 
                     + np.vstack(machine.phi0[turns]))
                    / (float(machine.h_num)
                       * np.vstack(machine.omega_rev0[turns])
                       * machine.dtbin) - xorigin)

    yp[profiles] = (tracked_denergy[profiles]
                    / float(dEbin) + machine.synch_part_y)
    return xp, yp

def ready_for_tomography(xp, yp, nbins):
    xp, yp, lost = filter_lost(xp, yp, nbins)
    log.info(f'number of lost particles: {lost}')
    xp = xp.astype(np.int32).T
    yp = yp.astype(np.int32).T

    return xp, yp

def _assert_coordinates(coordinates):
    if not hasattr(coordinates, '__iter__'): 
        raise expt.InvalidParticleError('coordinates should be itearble')
    if not len(coordinates) == 2:
        raise expt.InvalidParticleError('Two arrays of coordinates should be'
                                        'provided')
    for coord in coordinates:
        if not hasattr(coord, '__iter__'):
            raise expt.InvalidParticleError('coordinates should be itearble')
    if not len(coordinates[0]) == len(coordinates[1]):
         raise expt.InvalidParticleError(
                'arrays holding coordinates of x and y axis '
                'should have the same length')
    return coordinates[0], coordinates[1]