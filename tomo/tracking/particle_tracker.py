import numpy as np
from numba import njit 

class ParticleTracker:

    def __init__(self, time_space, mapinfo):
        self.timespace = time_space
        self.mapinfo = mapinfo

    def filter_lost_paricles(self, xp, yp):
        tpar = self.timespace.par
        nr_lost_pts = 0

        # Find all invalid particle values
        invalid_pts = np.argwhere(np.logical_or(xp >= tpar.profile_length,
                                                xp < 0))

        if np.size(invalid_pts) > 0:
            # Find all invalid particles
            invalid_pts = np.unique(invalid_pts.T[1])
            nr_lost_pts = len(invalid_pts)

            # Removing invalid particles
            xp = np.delete(xp, invalid_pts, axis=1)
            yp = np.delete(yp, invalid_pts, axis=1)

        return xp, yp, nr_lost_pts  

    def find_nr_of_particles(self):
        jdiff = (self.mapinfo.jmax
                 - self.mapinfo.jmin)

        pxls = np.sum(jdiff[self.mapinfo.imin
                            :self.mapinfo.imax + 1])
        return int(pxls * self.timespace.par.snpt**2)

    def _populate_bins(self, sqrtNbrPoints):
        xCoords = ((2.0 * np.arange(1, sqrtNbrPoints + 1) - 1)
                   / (2.0 * sqrtNbrPoints))
        yCoords = xCoords

        xCoords = xCoords.repeat(sqrtNbrPoints, 0).reshape(
            (sqrtNbrPoints, sqrtNbrPoints))
        yCoords = np.repeat([yCoords], sqrtNbrPoints, 0)
        return [xCoords, yCoords]

    # Wrapper function for creating of homogeneously distributed particles
    def _homogeneous_distribution(self):
        # Initializing points for homogeneous distr. particles
        points = self._populate_bins(self.timespace.par.snpt)
        nparts = self.find_nr_of_particles()
        xp = np.zeros(nparts)
        yp = np.copy(xp)

        # Creating the first profile with equally distributed points
        (xp,
         yp) = self._init_tracked_point(
                        self.timespace.par.snpt, self.mapinfo.imin,
                        self.mapinfo.imax, self.mapinfo.jmin,
                        self.mapinfo.jmax, xp,
                        yp, points[0], points[1])

        return xp, yp, nparts

    # Checks that the input arguments are correct, and spilts
    #  up to initial x and y coordnates. Also reads the start profile.
    @staticmethod
    def _manual_distribution(init_coords):
        correct = False
        if len(init_coords) == 2:
            in_xp = init_coords[0]
            in_yp = init_coords[1]
            if type(in_xp) is np.ndarray and type(in_yp) is np.ndarray: 
                if len(in_xp) == len(in_yp):
                    correct = True
        
        if not correct:
            err_msg = 'Unexpected amount of arguments.\n'\
                      'init_coords = (x, y, profile)\n'\
                      'x and y should be ndarrays of the same length, '\
                      'containing the inital values '\
                      'of the particles to be tracked.'
            raise AssertionError(err_msg)

        return in_xp, in_yp, len(in_xp)

    def rfv_at_turns(self):
        rf1v = (self.timespace.par.vrf1
                + self.timespace.par.vrf1dot
                * self.timespace.par.time_at_turn) * self.timespace.par.q
        rf2v = (self.timespace.par.vrf2
                + self.timespace.par.vrf2dot
                * self.timespace.par.time_at_turn) * self.timespace.par.q
        return rf1v, rf2v

    @staticmethod
    @njit
    # Creating homogeneously distributed particles
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

    @staticmethod
    def coords_to_physical(par, xp, yp, dEbin, turn=0):
        dphi = ((xp + par.x_origin)
                * par.h_num
                * par.omega_rev0[turn]
                * par.dtbin
                - par.phi0[turn])
        denergy = (yp - par.yat0) * dEbin
        return dphi, denergy
