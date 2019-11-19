import numpy as np

class ParticleTracker:

    # The tracking routine works on a copy of the input coordinates.
    def __init__(self, machine):
        self.machine = machine
        self.nturns = machine.dturns * (machine.nprofiles - 1)


    # Checks that the input arguments are correct, and spilts
    #  up to initial x and y coordnates. Also reads the start profile.
    def _assert_initial_parts(self, init_coords):
        correct = False
        if len(init_coords) == 2:
            in_xp = np.copy(init_coords[0])
            in_yp = np.copy(init_coords[1])
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
