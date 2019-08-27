import logging
import time as tm
import sys
import numpy as np
import matplotlib.pyplot as plt
from tracking import Tracking
from time_space import TimeSpace
from map_info import MapInfo
from new_tomo_cpp import NewTomographyC
from utils.assertions import TomoAssertions as ta

logging.basicConfig(level=logging.INFO)

def main():
    
    # Get input (and output if uncomment) path
    input_path = get_paths(live=True)
    
    # Collecting time space parameters and data
    ts = TimeSpace(input_path)

    # Setting path for all output as path read from file
    output_path = adjust_out_path(ts.par.output_dir)

    ts.save_profiles_text(ts.profiles[ts.par.filmstart-1, :],
                          output_path, f'profile{ts.par.filmstart:03d}.data')

    if ts.par.self_field_flag:
        ts.save_profiles_text(ts.vself[ts.par.filmstart-1,
                                       :ts.par.profile_length],
                              output_path, f'vself{ts.par.filmstart:03d}.data')
    
    # Creating map outlining for reconstruction
    mi = MapInfo(ts)

    # mi.write_jmax_tofile(ts, mi, output_path) # Don't know if used operationally
    mi.write_plotinfo_tofile(ts, mi, output_path)

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
    tomo = NewTomographyC(ts, xp, yp)
    weight = tomo.run_cpp()

    for film in range(ts.par.filmstart - 1, ts.par.filmstop, ts.par.filmstep):
        save_image(xp, yp, weight, ts.par.profile_length, film, output_path)
    
    save_difference(tomo.diff, output_path, ts.par.filmstart - 1)

def save_difference(diff, output_path, film):
    # Saving to file with numbers counting from one
    logging.info(f'Saving saving difference to {output_path}')
    np.savetxt(f'{output_path}d{film + 1:03d}.data', diff)

def save_image(xp, yp, weight, n_bins, film, output_path):
    phase_space = np.zeros((n_bins, n_bins))
    
    # Creating n_bins * n_bins phase-space image
    logging.info(f'Saving picture {film}.') 
    for x, y, w in zip(xp[:, film], yp[:, film], weight):
        phase_space[x, y] += w
    
    # Suppressing negative numbers
    phase_space = phase_space.clip(0.0)

    # Normalizing
    phase_space /= np.sum(phase_space)

    # Saving to file with numbers counting from one
    logging.info(f'Saving image{film} to {output_path}')
    # np.save(f'{output_path}py_image{film + 1:03d}', phase_space)
    np.savetxt(f'{output_path}image{film + 1:03d}.data',
               phase_space.flatten())

def get_paths(live):
    if live:
        try:
            input_path = sys.argv[1]
            # output_path = sys.argv[2]
        except IndexError:
            print('Error: You must provide an input file')
            print('Usage: main_testing <input_path>')
            sys.exit('Program exit..')
    else:
        input_path = '/afs/cern.ch/work/c/cgrindhe/tomography/input_v2.dat'
        # output_path = '/tmp/'
    
    # adjust_out_path(output_dir)

    return input_path # , output_path

# Making sure that output path ends on a dash
def adjust_out_path(output_dir):
    if output_dir[-1] != '/':
         output_dir += '/'
    return output_dir

    
main()