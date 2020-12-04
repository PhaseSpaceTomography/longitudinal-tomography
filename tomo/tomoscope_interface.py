import logging
import time as tm
import sys
import numpy as np
import matplotlib.pyplot as plt
from tracking import tracking
from time_space import TimeSpace
from map_info import MapInfo
from new_tomo_cpp import NewTomographyC
from utils.assertions import TomoAssertions as ta
from utils.exceptions import InputError

def main():
    
    raw_param, raw_data = get_input_file()
    
    print(' Start')   

    # Collecting time space parameters and data
    ts = TimeSpace()
    ts.create(raw_param, raw_data)

    # Deleting input data
    del(raw_param)
    del(raw_data)
    
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

    # mi.write_jmax_tofile(ts, mi, output_path) # Needed operationally?
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
    tomo = NewTomographyC(ts, xp, yp)
    weight = tomo.run_cpp()

    for film in range(ts.par.filmstart - 1, ts.par.filmstop, ts.par.filmstep):
        save_image(xp, yp, weight, ts.par.profile_length, film, output_path)
    
    save_difference(tomo.diff, output_path, ts.par.filmstart - 1)



def save_difference(diff, output_path, film):
    # Saving to file with numbers counting from one
    logging.info(f'Saving saving difference to {output_path}')
    # np.savetxt(f'{output_path}d{film + 1:03d}.data', diff) as f:
        
    with open(f'{output_path}d{film + 1:03d}.data', 'w') as f:
        for i, d in enumerate(diff):
            if i < 10:
                f.write(f'           {i}  {d:0.7E}\n')
            else:
                f.write(f'          {i}  {d:0.7E}\n')
                

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
    
    logging.info(f'Saving image{film} to {output_path}')
    # np.save(f'{output_path}py_image{film + 1:03d}', phase_space)
    # np.savetxt(f'{output_path}image{film + 1:03d}.data',
    #            phase_space.flatten())
    out_ps = phase_space.flatten()
    with open(f'{output_path}image{film + 1:03d}.data', 'w') as f:
        for element in out_ps:
            f.write(f'  {element:0.7E}\n')


def get_input_file(header_size=98, raw_data_file_idx=12):

#    read = list(sys.stdin)

    read = []
    finished = False
    lineNum = 0
    nDatPoints = 97
    while finished is False:
        read.append(sys.stdin.readline())
        if lineNum == 16:
            nFrames = int(read[-1])
        if lineNum == 20:
            nBins = int(read[-1])
            nDatPoints += nFrames*nBins
        if lineNum == nDatPoints:
            break
        lineNum += 1

    try:
        read_parameters = read[:header_size]
        for i in range(header_size):
            read_parameters[i] = read_parameters[i].strip('\r\n')
        if read_parameters[raw_data_file_idx] == 'pipe':
            read_data = np.array(read[header_size:], dtype=float)
        else:
            read_data = np.genfromtxt(read_parameters[raw_data_file_idx], dtype=float)
    except:
        raise InputError('The input file is not valid!')

    return read_parameters, read_data

def adjust_out_path(output_path):
    if output_path[-1] != '/':
        output_path += '/'
    return output_path

    
main()
