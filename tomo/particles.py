import numpy as np

from . import phase_space_info as psi
from .utils import assertions as asrt
from .utils import exceptions as expt


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
    def __init__(self, machine):
        self._machine = machine
        self._psinfo = psi.PhaseSpaceInfo(self._machine)
        self._psinfo.find_binned_phase_energy_limits()
        self.dEbin = self._psinfo.dEbin
        self.xorigin = self._psinfo.xorigin

        self.x_coords = None
        self.y_coords = None

    # The function wil create a homogeneous distribution of particles within
    # an area defined by the user. The area is given by the i and jlimits
    # found in the _psinfo object. Depending on the 'full_pp_flag' set in
    # the input parameters, the particles will be distribted in the bucket
    # area or the full image width.
    # This creates a particle distribution resembeling the original Fortran
    # version.
    # The particles coordinates will be saved as fractions of bins in
    # the x (phase) and y (energy) axis.
    def homogeneous_distribution(self):
        nbins_y = np.sum(self._psinfo.jmax[self._psinfo.imin:
                                            self._psinfo.imax + 1]
                        - self._psinfo.jmin[self._psinfo.imin:
                                             self._psinfo.imax + 1])

        # creating the distribution of particles within one cell.
        bin_pts = ((2.0 * np.arange(1, self._machine.snpt + 1) - 1)
                        / (2.0 * self._machine.snpt))

        # Creating x coordinates
        x = np.arange(self._psinfo.imin, self._psinfo.imax + 1, dtype=float)
        nbins_x = len(x)
        x = np.repeat(x, self._machine.snpt)
        x += np.tile(bin_pts, nbins_x)
        
        # Creating y coordinates.
        nbins_y = np.max(self._psinfo.jmax) - np.min(self._psinfo.jmin)
        y = np.arange(np.min(self._psinfo.jmin),
                      np.max(self._psinfo.jmax), dtype=float)
        y = np.repeat(y, self._machine.snpt)        
        y += np.tile(bin_pts, nbins_y)

        coords = np.meshgrid(x, y)
        coords = np.array([coords[0].flatten(), coords[1].flatten()])

        # Remove particles outside of the ijlimits.
        coords = coords[:,coords[1]
                          < self._psinfo.jmax[coords[0].astype(int)]]
        coords = coords[:, coords[1]
                          > self._psinfo.jmin[coords[0].astype(int)]]

        self.x_coords = coords[0]
        self.y_coords = coords[1]

    # Manually set the particles to be tracked.
    # The particles coordinates should be given in fractions of bins.
    def set_coordinates(self, in_x, in_y):
        if len(in_x) != len(in_y):
            raise AssertionError('Different shape of arrays containing '
                                 'x and y coordinates for particles')
        self.x_coords = in_x
        self.y_coords = in_y

    # Convert particle coordinates from coordinates as fractions of bins,
    # to physical units. The physical units are phase (x-axis),
    # and energy (y-axis).
    # This format is needed for the particle tracking routine.  
    def init_coords_to_physical(self, turn):
        dphi = ((self.x_coords + self.xorigin)
                * self._machine.h_num
                * self._machine.omega_rev0[turn]
                * self._machine.dtbin
                - self._machine.phi0[turn])
        denergy = (self.y_coords - self._machine.yat0) * self.dEbin
        return dphi, denergy

    # Convert from physical units to coordinates in bins.
    # Input is a two-dimensional array containing the tracked points.
    # The output is a two-dimensional aray containing the tracked
    # points given as nr of bins in phase and energy.
    def physical_to_coords(self, tracked_dphi, tracked_denergy):
        if tracked_dphi.shape != tracked_denergy.shape:
            raise AssertionError('Different shape of arrays containing '
                                 'phase and energies')
        nprof = tracked_denergy.shape[0]

        profiles = np.arange(nprof)
        turns = profiles * self._machine.dturns

        xp = np.zeros(tracked_dphi.shape)
        yp = np.zeros(tracked_dphi.shape)

        xp[profiles] = ((tracked_dphi[profiles] 
                         + np.vstack(self._machine.phi0[turns]))
                        / (float(self._machine.h_num)
                           * np.vstack(
                                self._machine.omega_rev0[turns])
                           * self._machine.dtbin)
                        - self.xorigin)

        yp[profiles] = (tracked_denergy[profiles] / float(self.dEbin)
                        + self._machine.yat0)
        return xp, yp


    def filter_lost_paricles(self, xp, yp):
        nr_lost_pts = 0

        # Find all invalid particle values
        invalid_pts = np.argwhere(
                        np.logical_or(
                            xp >= self._machine.nbins, xp < 0))
            
        if np.size(invalid_pts) > 0:
            # Find all invalid particles
            invalid_pts = np.unique(invalid_pts.T[1])
            nr_lost_pts = len(invalid_pts)
            # Removing invalid particles
            xp = np.delete(xp, invalid_pts, axis=1)
            yp = np.delete(yp, invalid_pts, axis=1)

        return xp, yp, nr_lost_pts

    def _assert_machine(self, machine):
        needed_fieds = ['snpt', 'xorigin', 'h_num', 
                        'omga_rev0', 'dtbin', 'phi0',
                        'yat0', 'dturns', 'nbins']
        asrt.assert_fields(machine, 'machine', needed_fieds,
                           expt.MachineParameterError,
                           'Did you remember to use machine.fill()?')
