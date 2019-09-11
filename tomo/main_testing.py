import logging
import time as tm
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tracking.tracking import Tracking
from time_space import TimeSpace
from map_info import MapInfo
from parameters import Parameters
# from tomography.tomography_py import TomographyPy
from tomography.tomography_cpp import TomographyCpp
from utils.assertions import TomoAssertions as ta
from utils.exceptions import InputError

logging.basicConfig(level=logging.INFO)

def main():
    
    raw_param, raw_data = get_input_from_file()

    parameter = Parameters()
    ts = TimeSpace()
    
    parameter.fill_from_array(raw_param)
    ts.create(parameter, raw_data)

    # Deleting input data
    del(raw_param)
    del(raw_data)
    
    output_path = adjust_outpath(ts.par.output_dir)

    ts.save_profiles_text(ts.profiles, output_path, 'py_profiles.dat')

    if ts.par.self_field_flag:
        ts.save_profiles_text(ts.vself[:, :ts.par.profile_length],
                              output_path, 'py_vself.dat')
    
    # Creating map outlining for reconstruction
    mi = MapInfo(ts)

    mi.write_jmax_tofile(ts, mi, output_path)
    mi.print_plotinfo()

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
        save_image(xp, yp, weight, ts.par.profile_length, film, output_path)

    save_difference(tomo.diff, output_path)

    print('Program finished.')


def save_coordinates(xp, yp, output_path):
    logging.info(f'Saving saving coordinates to {output_path}')
    logging.info('Saving xp')
    np.save(output_path + 'xp', xp)
    logging.info('Saving yp')
    np.save(output_path + 'yp', yp)


def save_difference(diff, output_path):
    logging.info(f'Saving saving difference to {output_path}')
    np.savetxt(f'{output_path}diff.dat', diff)


def save_image(xp, yp, weight, n_bins, film, output_path):
    phase_space = np.zeros((n_bins, n_bins))
    
    # Creating n_bins * n_bins phase-space image
    logging.info(f'Saving picture {film}.') 
    for x, y, w in zip(xp[:, film], yp[:, film], weight):
        phase_space[x, y] += w
    
    # Surpressing negative numbers
    phase_space = phase_space.clip(0.0)

    # Normalizing
    phase_space /= np.sum(phase_space)
    
    logging.info(f'Saving image{film} to {output_path}')
    np.save(f'{output_path}py_image{film}', phase_space)


def _get_input_args(output_dir_idx):
    input_file_pth = sys.argv[1]

    if not os.path.isfile(input_file_pth):
        raise InputError(f'The input file: "{input_file_pth}" '
                         f'does not exist!')

    with open(input_file_pth, 'r') as f:
        read = f.readlines()

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        if os.path.isdir(output_dir):
            read[output_dir_idx] = output_dir
        else:
            raise InputError(f'The chosen output directory: "{output_dir}" '
                             f'does not exist!')
    return read


def _get_input_stdin():
    return list(sys.stdin)

# Mabye change to calculate how many data points there should be from parameters,
#   and then check if the file has the right size
def _split_input(read_input, header_size, raw_data_file_idx):
    try:
        read_parameters = read_input[:header_size]
        for i in range(header_size):
            read_parameters[i] = read_parameters[i].strip('\r\n')

        if read_parameters[raw_data_file_idx] == 'pipe':
            read_data = np.array(read_input[header_size:], dtype=float)
        else:
            read_data = np.genfromtxt(read_parameters[raw_data_file_idx],
                                      dtype=float)
    except:
        raise InputError('Something went wrong when reading the input.')
    
    return read_parameters, read_data


def get_input_from_file(header_size=98,
                        raw_data_file_idx=12, output_dir_idx=14):
    if len(sys.argv) > 1:
        read = _get_input_args(output_dir_idx)
    else:
        read = _get_input_stdin()

    return _split_input(read, header_size, raw_data_file_idx)


def adjust_outpath(output_path):
    if output_path[-1] != '/':
        output_path += '/'
    return output_path

    
main()