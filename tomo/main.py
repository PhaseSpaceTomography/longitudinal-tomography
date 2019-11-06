# Not in use, but handy to keep for now...
import time as tm
import sys
import matplotlib.pyplot as plt
import os
# --/ end \--

import logging
import numpy as np
# from tracking.tracking import Tracking
from tracking.tracking_cpp import TrackingCpp
from time_space import TimeSpace
from map_info import MapInfo
from parameters import Parameters
from tomography.tomography_cpp import TomographyCpp
from utils.exceptions import InputError
from utils.assertions import TomoAssertions as ta
from utils.tomo_io import InputHandler, OutputHandler

logging.basicConfig(level=logging.INFO)

def main(*args):
    
    print('Start')

    # Case: only parameter is given
    #  The file containing the measured datamust in this case be
    #  separate from the parameters file, and contain measurement data ONLY.
    if len(args) == 1:
        if type(args[0]) is Parameters:
            parameter = args[0]
            parameter.fill()
            if os.path.isfile(parameter.rawdata_file):
                raw_data = np.genfromtxt(parameter.rawdata_file)
            else:
                err_msg = f'The given data file: "{parameter.rawdata_file}" '\
                          f'does not exist.'
                raise SystemError(err_msg)
        else:
            raise InputError('Given argument should be of type "Parameters".')
    # In this case the raw data is given as input argument.
    # The raw data wil not be re-loaded.
    elif len(args) == 2:
        if type(args[0]) is Parameters and type(args[1]) is np.ndarray:
            parameter = args[0]
            parameter.fill()
            raw_data = args[1]
        elif type(args[1]) is Parameters and type(args[0]) is np.ndarray:
            parameter = args[1]
            parameter.fill()
            raw_data = args[0]
        else:
            raise InputError('Arguments should be of type '\
                             '"Parameters" and "ndarray"')
    else:
        err_msg = f'The wrong nr of arguments were given.\n'\
                  f'Aguments recieved: {len(args)}, valid number of '\
                  f'arguments is one or two.'
        raise InputError(err_msg)

    ts = TimeSpace(parameter)
    ts.create(raw_data)
    
    output_path = OutputHandler.adjust_outpath(ts.par.output_dir)

    OutputHandler.save_profile_ccc(ts.profiles, ts.par.filmstart-1,
                                   output_path)

    if ts.par.self_field_flag:
        OutputHandler.save_self_volt_profile_ccc(
                        ts.vself[:, :ts.par.profile_length], output_path)

    # Creating map outlining for reconstruction
    mi = MapInfo(ts)
    mi.find_ijlimits()
    mi.print_plotinfo_ccc(ts)

    # Particle tracking
    # tr = Tracking(ts, mi)
    tr = TrackingCpp(ts, mi)
    xp, yp = tr.track()

    ta.assert_only_valid_particles(xp, ts.par.profile_length)

    # Transposing needed for tomography routine
    # -1 is Fortran compensation (now counting from 0)
    # OBS: This takes a notable amount of time
    #      (~0.5s for C500MidPhaseNoise)
    xp = np.ceil(xp).astype(int).T - 1
    yp = np.ceil(yp).astype(int).T - 1
    
    # Reconstructing phase space  
    tomo = TomographyCpp(ts, xp, yp)
    # weight = tomo.run()
    weight = tomo.run_cpp()

    for film in range(ts.par.filmstart - 1, ts.par.filmstop, ts.par.filmstep):
        image = TomographyCpp.create_phase_space_image(
                        xp, yp, weight, ts.par.profile_length, film)
        OutputHandler.save_phase_space_ccc(image, film, output_path)

    OutputHandler.save_difference_ccc(tomo.diff, output_path, film)

    return image, tomo.diff, ts.profiles[film] 


if __name__ == "__main__": 
    raw_param, raw_data = InputHandler.get_input_from_file()
    parameter = Parameters()
    parameter.parse_from_txt(raw_param)
    main(parameter, raw_data)
