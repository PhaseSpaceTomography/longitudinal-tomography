import numpy as np
from numba import njit          # Needed for fortran style init.
from map_info import MapInfo
import sys
import time as tm

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
    def __init__(self, timespace):
        self._timespace = timespace
        self._mapinfo = MapInfo(self._timespace)
        self._mapinfo.find_ijlimits()
        self.dEbin = self._mapinfo.dEbin

        self.x_coords = None
        self.y_coords = None

    # Use the ff-flag for giving a Fortran style output needed for the
    # tomoscope application.
    # The function wil create a homogeneous distribution of particles within
    # an area defined by the user. The area is given by the i and jlimits
    # found in the _mapinfo object. Depending on the 'full_pp_flag' set in
    # the input parameters, the particles will be distribted in the bucket
    # area or the full image width.
    # This creates a particle distribution resembeling the original Fortran
    # version.
    # The particles coordinates will be saved as fractions of bins in
    # the x (phase) and y (energy) axis.
    def homogeneous_distribution(self, ff=False):
        if ff:
            self._mapinfo.print_plotinfo_ccc(self._timespace)

        nbins_y = np.sum(self._mapinfo.jmax[self._mapinfo.imin:
                                            self._mapinfo.imax + 1]
                         - self._mapinfo.jmin[self._mapinfo.imin:
                                              self._mapinfo.imax + 1])

        # creating the distribution of particles within one cell.
        parts_in_bin = ((2.0 * np.arange(1, self._timespace.par.snpt + 1) - 1)
                        / (2.0 * self._timespace.par.snpt))

        # Creating x coordinates
        nbins_x = self._mapinfo.imax - self._mapinfo.imin
        x = np.arange(self._mapinfo.imin, self._mapinfo.imax, dtype=float)
        x = np.repeat(x, self._timespace.par.snpt)
        x += np.tile(parts_in_bin, nbins_x)
        
        # Creating y coordinates.
        nbins_y = np.max(self._mapinfo.jmax) - np.min(self._mapinfo.jmin)
        y = np.arange(np.min(self._mapinfo.jmin),
                      np.max(self._mapinfo.jmax), dtype=float)
        y = np.repeat(y, self._timespace.par.snpt)        
        y += np.tile(parts_in_bin, nbins_y)

        coords = np.meshgrid(x, y)
        coords = np.array([coords[0].flatten(), coords[1].flatten()])

        # Remove particles outside of the ijlimits.
        coords = coords[:,coords[1]
                          < self._mapinfo.jmax[coords[0].astype(int)]]
        coords = coords[:, coords[1]
                          > self._mapinfo.jmin[coords[0].astype(int)]]

        self.x_coords = coords[0]
        self.y_coords = coords[1]

    # Manually set the particles to be tracked.
    # The particles coordinates should be given in fractions of bins.
    def set_coordinates(self, in_x, in_y):
        if len(in_x) != len(in_y):
            raise AssertionError('Different shape of arrays containing '
                                 'x and y coordinates')
        self.x_coords = in_x
        self.y_coords = in_y

    # Convert particle coordinates from coordinates as fractions of bins,
    # to physical units. The physical units are phase (x-axis),
    # and energy (y-axis).
    # This format is needed for the particle tracking routine.  
    def init_coords_to_physical(self, turn):
        dphi = ((self.x_coords + self._timespace.x_origin)
                * self._timespace.par.h_num
                * self._timespace.par.omega_rev0[turn]
                * self._timespace.par.dtbin
                - self._timespace.par.phi0[turn])
        denergy = (self.y_coords - self._timespace.par.yat0) * self.dEbin
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
        turns = profiles * self._timespace.par.dturns

        xp = np.zeros(tracked_dphi.shape)
        yp = np.zeros(tracked_dphi.shape)

        xp[profiles] = ((tracked_dphi[profiles] 
                         + np.vstack(self._timespace.par.phi0[turns]))
                        / (float(self._timespace.par.h_num)
                           * np.vstack(self._timespace.par.omega_rev0[turns])
                           * self._timespace.par.dtbin)
                        - self._timespace.x_origin)

        yp[profiles] = (tracked_denergy[profiles] / float(self._mapinfo.dEbin)
                        + self._timespace.par.yat0)

        return xp, yp

    # =========================== OLD ROUTINES ===============================
    # ------------------ Fortran style particle initialization ---------------
    # To be deleted in future.
    # Much slower and particles are araanged in a different way.
    # ------------------------------------------------------------------------
    def fortran_homogeneous_distribution(self, timespace):
        self._mapinfo = MapInfo(timespace)
        self._mapinfo.find_ijlimits()
        self.dEbin = self._mapinfo.dEbin
        
        points = self._populate_bins(timespace.par.snpt)
        nparts = self._find_nr_of_particles(timespace.par.snpt)
        
        xp = np.zeros(nparts)
        yp = np.copy(xp)

        # Creating the first profile with equally distributed points
        (xp,
         yp) = self._init_tracked_point(
                        timespace.par.snpt, self._mapinfo.imin,
                        self._mapinfo.imax, self._mapinfo.jmin,
                        self._mapinfo.jmax, xp,
                        yp, points[0], points[1])

        return xp, yp, nparts

    def _populate_bins(self, snpt):
        xCoords = ((2.0 * np.arange(1, snpt + 1) - 1)
                   / (2.0 * snpt))
        yCoords = xCoords

        xCoords = xCoords.repeat(snpt, 0).reshape((snpt, snpt))
        yCoords = np.repeat([yCoords], snpt, 0)
        return [xCoords, yCoords]

    def _find_nr_of_particles(self, snpt):
        jdiff = (self._mapinfo.jmax
                 - self._mapinfo.jmin)

        pxls = np.sum(jdiff[self._mapinfo.imin
                            :self._mapinfo.imax + 1])
        return int(pxls * snpt**2)

    @staticmethod
    @njit
    def _init_tracked_point(snpt, imin, imax,
                            jmin, jmax, xp, yp,
                            xpoints, ypoints):
        k = 0
        for iLim in range(imin, imax + 1):
            for jLim in range(jmin[iLim], jmax[iLim]):
                for i in range(snpt):
                    for j in range(snpt):
                        xp[k] = iLim + xpoints[i, j]
                        yp[k] = jLim + ypoints[i, j]
                        k += 1
        return xp, yp

    # ---------------- End Fortran style particle initialization -----------------

    # The function which was called for each profile measurement turn in
    #  the old particle tracking. Kept for reference.
    def fortran_physical_to_coords(self):
        raise NotImplementedError('Not implemented, only kept for refere')
        xp[profiles] = ((dphi + timespace.par.phi0[turn])
                       / (float(timespace.par.h_num)
                       * timespace.par.omega_rev0[turn]
                       * timespace.par.dtbin)
                       - timespace.x_origin)
        yp[profiles] = denergy / float(dEbin) + timespace.yat0
    # ======================== END OLD ROUTINES ============================
