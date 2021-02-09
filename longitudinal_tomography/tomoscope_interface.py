import logging
import time as tm
import sys
import numpy as np
import matplotlib.pyplot as plt
from .exceptions import InputError
from tracking import tracking as tracking
import tomography.tomography as tomography
import utils.data_treatment as dtreat
import utils.tomo_input as tomoin
import utils.tomo_output as tomoout
import tracking.particles as pts

def main():

    raw_param, raw_data = get_input_file()

    print(' Start')
    # Generating machine object
    machine, frames = tomoin.txt_input_to_machine(raw_param)
    machine.values_at_turns()
    waterfall = frames.to_waterfall(raw_data)

    # Setting path for all output as path read from file
    output_path = machine.output_dir
    if output_path[-1] != '/':
        output_path += ('/')

    # Creating profiles object
    profiles = tomoin.raw_data_to_profiles(
                    waterfall, machine, frames.rebin, frames.sampling_time)
    profiles.calc_profilecharge()

    if profiles.machine.synch_part_x < 0:
        fit_info = dtreat.fit_synch_part_x(profiles)
        machine.load_fitted_synch_part_x_ftn(fit_info)
    reconstr_idx = machine.filmstart

    # Tracking...
    tracker = tracking.Tracking(machine)
    tracker.enable_fortran_output(profiles.profile_charge)

    if tracker.self_field_flag:
        profiles.calc_self_fields()
        tracker.enable_self_fields(profiles)

    # Particle tracking
    xp, yp = tracker.track(reconstr_idx)

    # Converting from physical coordinates ([rad], [eV])
    # to phase space coordinates.
    if not tracker.self_field_flag:
        xp, yp = pts.physical_to_coords(
                    xp, yp, machine, tracker.particles.xorigin,
                    tracker.particles.dEbin)

    # Filter out lost particles, transposes particle matrix, casts to np.int32.
    xp, yp = pts.ready_for_tomography(xp, yp, machine.nbins)

    # Tomography!
    tomo = tomography.TomographyCpp(profiles.waterfall, xp)
    weight = tomo.run(verbose=True)

    for film in range(machine.filmstart - 1, machine.filmstop,
                      machine.filmstep):
        save_image(xp, yp, weight, machine.nbins, film, output_path)

    save_difference(tomo.diff, output_path, machine.filmstart - 1)


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

if __name__ == '__main__':
    main()
