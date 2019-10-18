# Not in use, but handy to keep for now...
import time as tm
import sys
import matplotlib.pyplot as plt
# --/ end \--

import logging
import numpy as np
from tracking.tracking import Tracking
from time_space import TimeSpace
from map_info import MapInfo
from parameters import Parameters
from tomography.tomography_cpp import TomographyCpp
from utils.assertions import TomoAssertions as ta
from utils.tomo_io import InputHandler, OutputHandler

logging.basicConfig(level=logging.INFO)

def main():
    
    print('Start')

    raw_param, raw_data = InputHandler.get_input_from_file()

    parameter = Parameters()
    ts = TimeSpace()
    
    parameter.fill_from_array(raw_param)
    ts.create(parameter, raw_data)

    # Deleting input data
    del(raw_param)
    del(raw_data)
    
    output_path = OutputHandler.adjust_outpath(ts.par.output_dir)

    OutputHandler.save_profile_ccc(ts.profiles, ts.par.filmstart-1,
                                   output_path)

    if ts.par.self_field_flag:
        ts.save_profiles_text(ts.vself[:, :ts.par.profile_length],
                              output_path, 'py_vself.dat')
    
    # Creating map outlining for reconstruction
    mi = MapInfo(ts)

    mi.write_jmax_tofile(ts, mi, output_path)
    mi.print_plotinfo_ccc()

    # Particle tracking
    tr = Tracking(ts, mi)
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
    weight = tomo.run()

    for film in range(ts.par.filmstart - 1, ts.par.filmstop, ts.par.filmstep):
        OutputHandler.save_phase_space_ccc(xp, yp, weight,
                                           ts.par.profile_length,
                                           film, output_path)

    OutputHandler.save_difference_ccc(tomo.diff, output_path, film)
    
main()