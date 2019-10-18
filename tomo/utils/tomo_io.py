import os
import sys
import numpy as np
import logging
from utils.exceptions import InputError

class InputHandler:

    # TO BE ADDED:
    # - Safe reading from stdin with counting of data elements
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

    @classmethod
    def adjust_outpath(cls, output_path):
        if output_path[-1] != '/':
            output_path += '/'
        return output_path

    @classmethod
    def save_phase_space_npy(cls, xp, yp, weight, n_bins, film, output_path):
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

    @classmethod
    def save_difference_txt(cls, diff, output_path):
        logging.info(f'Saving saving difference to {output_path}')
        np.savetxt(f'{output_path}diff.dat', diff)    

    @classmethod
    def save_coordinates_npy(cls, xp, yp, output_path):
        logging.info(f'Saving saving coordinates to {output_path}')
        logging.info('Saving xp')
        np.save(output_path + 'xp', xp)
        logging.info('Saving yp')
        np.save(output_path + 'yp', yp)
