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
    def _initiate_points(self):
        # Initializing points for homogeneous distr. particles
        points = self._populate_bins(self.timespace.par.snpt)

        xp = np.zeros(self.find_nr_of_particles())
        yp = np.copy(xp)

        # Creating the first profile with equally distributed points
        (xp,
         yp) = self._init_tracked_point(
                        self.timespace.par.snpt, self.mapinfo.imin,
                        self.mapinfo.imax, self.mapinfo.jmin,
                        self.mapinfo.jmax, xp,
                        yp, points[0], points[1])

        return xp, yp

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