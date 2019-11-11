import numpy as np

class ParticleTracker:

    # def __init__(self, time_space, mapinfo):
    #     self.timespace = time_space
    #     self.mapinfo = mapinfo

    def __init__(self, parameter):
        self.parameter = parameter
        self.nturns = parameter.dturns * (parameter.profile_count - 1)

    # def filter_lost_paricles(self, xp, yp):
    #     tpar = self.timespace.par
    #     nr_lost_pts = 0

    #     # Find all invalid particle values
    #     invalid_pts = np.argwhere(np.logical_or(xp >= tpar.profile_length,
    #                                             xp < 0))

    #     if np.size(invalid_pts) > 0:
    #         # Find all invalid particles
    #         invalid_pts = np.unique(invalid_pts.T[1])
    #         nr_lost_pts = len(invalid_pts)

    #         # Removing invalid particles
    #         xp = np.delete(xp, invalid_pts, axis=1)
    #         yp = np.delete(yp, invalid_pts, axis=1)

    #     return xp, yp, nr_lost_pts

    # Checks that the input arguments are correct, and spilts
    #  up to initial x and y coordnates. Also reads the start profile.
    @staticmethod
    def _assert_initial_parts(init_coords):
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
        rf1v = (self.parameter.vrf1
                + self.parameter.vrf1dot
                * self.parameter.time_at_turn) * self.parameter.q
        rf2v = (self.parameter.vrf2
                + self.parameter.vrf2dot
                * self.parameter.time_at_turn) * self.parameter.q
        return rf1v, rf2v
