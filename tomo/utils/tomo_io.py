import os
import sys
import numpy as np
import logging as log
from utils.exceptions import InputError

class InputHandler:

    # TO BE ADDED:
    # - assertions that the file is correct etc... 

    @classmethod
    def get_input_from_file(cls, header_size=98,
                            raw_data_file_idx=12, output_dir_idx=14):
        if len(sys.argv) > 1:
            read = cls._get_input_args(output_dir_idx)
        else:
            read = cls._get_input_stdin()
    
        return cls._split_input(read, header_size, raw_data_file_idx)

    @classmethod
    def _get_input_stdin(cls):
        read = []
        finished = False
        line_num = 0
        ndata_points = 97
        while not finished:
            print(line_num)
            read.append(sys.stdin.readline())
            if line_num == 16:
                nframes = int(read[-1])
            if line_num == 20:
                nbins = int(read[-1])
                ndata_points += nframes*nbins
            if line_num == ndata_points:
                finished = True
            line_num += 1
        return read


    @classmethod
    def _get_input_args(cls, output_dir_idx):
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
                raise InputError(f'The chosen output directory: '
                                 f'"{output_dir}" does not exist!')
        return read

    # Mabye change to calculate how many data points there should be
    #  from parameters, and then check if the file has the right size
    @classmethod
    def _split_input(cls, read_input, header_size, raw_data_file_idx):
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
            raise InputError('Something went wrong while reading the input.')
        
        return read_parameters, read_data


class OutputHandler:
    
    # TO BE ADDED:
    # - saving all outut in the same way as fortran

    # --------------------------------------------------------------- #
    #                         UTILITIES                               #
    # --------------------------------------------------------------- #

    @classmethod
    # Assert that output path ends on a '/'
    def adjust_outpath(cls, output_path):
        if output_path[-1] != '/':
            output_path += '/'
        return output_path

    # --------------------------------------------------------------- #
    #                           PROFILES                              #
    # --------------------------------------------------------------- #

    @classmethod
    # Write phase space image to text-file in tomoscope format.
    def save_profile_ccc(cls, profiles, prof_idx, output_dir):
        out_profile = profiles[prof_idx].flatten()
        file_path = f'{output_dir}/profile{prof_idx + 1:03d}.data'
        with open(file_path, 'w') as f:
            for element in out_profile:    
                f.write(f' {element:0.7E}\n')

    # --------------------------------------------------------------- #
    #                         PHASE-SPACE                             #
    # --------------------------------------------------------------- #

    @classmethod
    # Write phase space image to .npy file.
    def save_phase_space_npy(cls, xp, yp, weight, n_bins, film, output_path):
        log.info(f'Saving image{film} to {output_path}')
        phase_space = cls.create_phase_space_image(xp, yp, weight,
                                                   n_bins, film)
        np.save(f'{output_path}image{film + 1:03d}', phase_space)

    @classmethod
    # Write phase space image to text-file in tomoscope format.
    def save_phase_space_ccc(cls, xp, yp, weight, n_bins, film, output_path):
        log.info(f'Saving image{film} to {output_path}')
        phase_space = cls.create_phase_space_image(xp, yp, weight,
                                                   n_bins, film)
        phase_space = phase_space.flatten()
        with open(f'{output_path}image{film + 1:03d}.data', 'w') as f:
            for element in phase_space:
                f.write(f'  {element:0.7E}\n')

    @classmethod
    # To be moved to tomography class?
    def create_phase_space_image(cls, xp, yp, weight, n_bins, film):
        phase_space = np.zeros((n_bins, n_bins))
    
        # Creating n_bins * n_bins phase-space image
        log.info(f'Saving picture {film}.')
        for x, y, w in zip(xp[:, film], yp[:, film], weight):
            phase_space[x, y] += w
    
        # Surpressing negative numbers
        phase_space = phase_space.clip(0.0)

        # Normalizing
        phase_space /= np.sum(phase_space)

        return phase_space

    # --------------------------------------------------------------- #
    #                         DISCREPANCY                             #
    # --------------------------------------------------------------- #
    
    @classmethod
    # Write difference to text file in tomoscope format
    def save_difference_ccc(cls, diff, output_path, film):
        # Saving to file with numbers counting from one
        log.info(f'Saving saving difference to {output_path}')
        with open(f'{output_path}d{film + 1:03d}.data', 'w') as f:
            for i, d in enumerate(diff):
                if i < 10:
                    f.write(f'           {i}  {d:0.7E}\n')
                else:
                    f.write(f'          {i}  {d:0.7E}\n')

    # --------------------------------------------------------------- #
    #                         COORDINATES                             #
    # --------------------------------------------------------------- #

    @classmethod
    def save_coordinates_npy(cls, xp, yp, output_path):
        log.info(f'Saving saving coordinates to {output_path}')
        log.info('Saving xp')
        np.save(output_path + 'xp', xp)
        log.info('Saving yp')
        np.save(output_path + 'yp', yp)
