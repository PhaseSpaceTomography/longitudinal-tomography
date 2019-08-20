import logging
import time as tm
import sys
import numpy as np
from tracking import Tracking
from time_space import TimeSpace
from map_info import MapInfo
from new_tomo_cpp import NewTomographyC
from utils.assertions import TomoAssertions as ta

logging.basicConfig(level=logging.INFO)

def main():
    
    input_path, output_path = get_paths(live=True)
    
    # Collecting time space parameters and data
    t0 = tm.perf_counter()
    ts = TimeSpace(input_path)
    time_ts = tm.perf_counter() - t0

    ts.save_profiles_text(ts.profiles, output_path, 'py_profiles.dat')

    if ts.par.self_field_flag:
        ts.save_profiles_text(ts.vself[:, :ts.par.profile_length],
                              output_path, 'py_vself.dat')
    
    # Creating map outlining for reconstruction
    t0 = tm.perf_counter()
    mi = MapInfo(ts)
    time_maps = tm.perf_counter() - t0

    mi.write_jmax_tofile(ts, mi, output_path)
    mi.write_plotinfo_tofile(ts, mi, output_path)

    # Particle tracking
    t0 = tm.perf_counter()
    tr = Tracking(ts, mi)
    xp, yp = tr.track()
    time_tracking = tm.perf_counter() - t0

    ta.assert_only_valid_particles(xp, ts.par.profile_length)

    # Transposing needed for tomography routine
    t0 = tm.perf_counter()
    xp = np.ceil(xp).astype(int).T
    yp = np.ceil(yp).astype(int).T
    time_transp = tm.perf_counter() - t0
    
    # Reconstructing phase space
    t0 = tm.perf_counter()   
    tomo = NewTomographyC(ts, xp, yp)
    weight = tomo.run_cpp()
    time_rec = tm.perf_counter() - t0

    # Writing summary of time usage
    print(f'Time - time space: {time_ts}s')
    print(f'Time - map creation: {time_maps}s')
    print(f'Time - tracking: {time_tracking}s')
    print(f'Time - Transposing: {time_transp}s')
    print(f'Time - time recreation: {time_rec}s')

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
    logging.info(f'Saving saving differennce to {output_path}')
    np.savetxt(f'{output_path}diff.dat', diff)

def save_image(xp, yp, weight, n_bins, film, output_path):
    phase_space = np.zeros((n_bins, n_bins))
    
    # Creating n_bins * n_bins phase-space image  
    for x, y, w in zip(xp[:, film], yp[:, film], weight):
        phase_space[x, y] += w
    
    # Normalizing
    phase_space /= np.sum(phase_space)

    # Surpressing negative numbers
    phase_space = phase_space.clip(0.0)

    logging.info(f'Saving image{film} to {output_path}')
    np.save(f'{output_path}py_image{film}', phase_space)

def get_paths(live):
    if live:
        try:
            input_path = sys.argv[1]
            output_path = sys.argv[2]
        except IndexError:
            print('Error: You must provide an input file and an output directory')
            print('Usage: main_testing <input_path> <output_path>')
            sys.exit('Program exit..')
    else:
        input_path = '/afs/cern.ch/work/c/cgrindhe/tomography/input_v2.dat'
        output_path = '/afs/cern.ch/work/c/cgrindhe/tomography/out'
    
    # Making sure that output path ends on a dash
    if output_path[-1] != '/':
        output_path += '/'

    return input_path, output_path
    
main()