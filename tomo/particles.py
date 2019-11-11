import numpy as np
from numba import njit # Needed for fortran style init.
from map_info import MapInfo
import sys
import time as tm

class Particles(object):

    def __init__(self):
        self.x = None
        self.y = None
        self.dEbin = None
        self._timespace = None
        self._mapinfo = None

    # Experimental.
    # ff: Fortran format 
    def homogeneous_distribution(self, timespace, ff=False):
        self._timespace = timespace 
        self._mapinfo = MapInfo(self._timespace)
        self._mapinfo.find_ijlimits()
        self.dEbin = self._mapinfo.dEbin

        if ff:
            self._mapinfo.print_plotinfo_ccc(self._timespace)

        nbins_y = np.sum(self._mapinfo.jmax[self._mapinfo.imin:
                                           self._mapinfo.imax + 1]
                        - self._mapinfo.jmin[self._mapinfo.imin:
                                             self._mapinfo.imax + 1])

        # To ensure that every bin contains the equal amount of equally
        #  spaced particles
        parts_in_bin = ((2.0 * np.arange(1, self._timespace.par.snpt + 1) - 1)
                        / (2.0 * self._timespace.par.snpt))

        # Creating x coordinates
        nbins_x = self._mapinfo.imax - self._mapinfo.imin
        x = np.arange(self._mapinfo.imin, self._mapinfo.imax, dtype=float)
        x = np.repeat(x, self._timespace.par.snpt)
        x += np.tile(parts_in_bin, nbins_x)
        
        # Creating y coordinates
        nbins_y = np.max(self._mapinfo.jmax) - np.min(self._mapinfo.jmin)
        y = np.arange(np.min(self._mapinfo.jmin),
                      np.max(self._mapinfo.jmax), dtype=float)
        y = np.repeat(y, self._timespace.par.snpt)        
        y += np.tile(parts_in_bin, nbins_y)

        coords = np.meshgrid(x, y)
        coords = np.array([coords[0].flatten(), coords[1].flatten()])

        coords = coords[:,coords[1]
                          < self._mapinfo.jmax[coords[0].astype(int)]]
        coords = coords[:, coords[1]
                          > self._mapinfo.jmin[coords[0].astype(int)]]

        self.x = coords[0]
        self.y = coords[1]

    # manually setting the x and y coordinates.
    def set_coordinates(self, x, y, timespace):
        if len(x) != len(y):
            raise AssertionError('Different shape of arrays containing '
                                 'x and y coordinates')
        self._timespace = timespace 
        self._mapinfo = MapInfo(self._timespace)
        self.dEbin = self._mapinfo.find_dEbin()
        self.x = x
        self.y = y

    # For converting from initial coordinates in phase-space to physical units 
    def init_coords_to_physical(self, turn):
        dphi = ((self.x + self._timespace.x_origin)
                * self._timespace.par.h_num
                * self._timespace.par.omega_rev0[turn]
                * self._timespace.par.dtbin
                - self._timespace.par.phi0[turn])
        denergy = (self.y - self._timespace.par.yat0) * self.dEbin
        return dphi, denergy

    # For converting from physical units to coordinates in phase-space
    @staticmethod
    def physical_to_coords(dphi, denergy, timespace):
        if dphi.shape != denergy.shape:
            raise AssertionError('Different shape of arrays containing '
                                 'phase and energies')

        mapinfo = MapInfo(timespace)
        dEbin = mapinfo.find_dEbin()

        nprof = denergy.shape[0]

        profiles = np.arange(nprof)
        turns = profiles * timespace.par.dturns

        xp = np.zeros(dphi.shape)
        yp = np.zeros(dphi.shape)

        xp[profiles] = ((dphi[profiles] 
                         + np.vstack(timespace.par.phi0[turns]))
                        / (float(timespace.par.h_num)
                           * np.vstack(timespace.par.omega_rev0[turns])
                           * timespace.par.dtbin)
                        - timespace.x_origin)

        yp[profiles] = denergy[profiles] / float(dEbin) + timespace.par.yat0

        return xp, yp

    # ------------------ Fortran style particle initialization -----------------
    # To be deleted in future.
    # Much slower and particles are araanged in a different way.
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
