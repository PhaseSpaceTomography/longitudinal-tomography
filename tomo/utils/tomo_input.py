import numpy as np
import logging as log
import os
import sys

from machine import Machine
from utils.exceptions import InputError

# Some constants for the input file containing machine parameters
PARAMETER_LENGTH = 98
RAW_DATA_FILE_IDX = 12
OUTPUT_DIR_IDX = 14

# Function to be called from main.
# Lets the user give input using stdin or via args
def get_user_input():
    if len(sys.argv) > 1:
        read = _get_input_args()
    else:
        read = _get_input_stdin()
    
    return _split_input(read)


# Recieve path to input file via sys.argv.
# Can also recieve the path to the output directory.
def _get_input_args():
    input_file_pth = sys.argv[1]
    
    if not os.path.isfile(input_file_pth):
        raise InputError(f'The input file: "{input_file_pth}" '
                         f'does not exist!')
    
    with open(input_file_pth, 'r') as f:
        read = f.readlines()
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        if os.path.isdir(output_dir):
            read[OUTPUT_DIR_IDX] = output_dir
        else:
            raise InputError(f'The chosen output directory: '
                             f'"{output_dir}" does not exist!')
    return read


# Read machine parameters via stdin.
# Here the measured data must be pipelined in the same file as
# the machine parameters.
def _get_input_stdin():
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
            ndata_points += nframes * nbins
        if line_num == ndata_points:
            finished = True
        line_num += 1
    return read


# Splits the read input data to machine parameters and raw data.
# If the raw data is not already read from the input file, the
#  data will be found in the file given by the parameter file.
def _split_input(read_input):
    try:
        read_parameters = read_input[:PARAMETER_LENGTH]
        for i in range(PARAMETER_LENGTH):
            read_parameters[i] = read_parameters[i].strip('\r\n')
        if read_parameters[RAW_DATA_FILE_IDX] == 'pipe':
            read_data = np.array(read_input[PARAMETER_LENGTH:], dtype=float)
        else:
            read_data = np.genfromtxt(read_parameters[RAW_DATA_FILE_IDX],
                                      dtype=float)
    except:
        raise InputError('Something went wrong while loading the input.')
        
    return read_parameters, read_data


# Function to convert from array containing the lines in an input file
#  to a partially filled machine object.
# The array must contain a direct read from an input file.
# TODO: Conversion from Fortran to python indexing.
def input_to_machine(input_array):
    if len(input_array) != PARAMETER_LENGTH:
        raise InputError

    for i in range(len(input_array)):
            input_array[i] = input_array[i].strip('\r\n')

    machine = Machine()
    machine.rawdata_file = input_array[12]
    machine.output_dir = input_array[14]
    machine.framecount = int(input_array[16])
    machine.frame_skipcount = int(input_array[18])
    machine.framelength = int(input_array[20])
    machine.dtbin = float(input_array[22])
    machine.dturns = int(input_array[24])
    machine.preskip_length = int(input_array[26])
    machine.postskip_length = int(input_array[28])
    machine.imin_skip = int(input_array[31])
    machine.imax_skip = int(input_array[34])
    machine.rebin = int(input_array[36])
    machine._xat0 = float(input_array[39])
    machine.demax = float(input_array[41])
    machine.filmstart = int(input_array[43])
    machine.filmstop = int(input_array[45])
    machine.filmstep = int(input_array[47])
    machine.niter = int(input_array[49])
    machine.snpt = int(input_array[51])
    machine.full_pp_flag = bool(int(input_array[53]))
    machine.beam_ref_frame = int(input_array[55])
    machine.machine_ref_frame = int(input_array[57])
    machine.vrf1 = float(input_array[61])
    machine.vrf1dot = float(input_array[63])
    machine.vrf2 = float(input_array[65])
    machine.vrf2dot = float(input_array[67])
    machine.h_num = float(input_array[69])
    machine.h_ratio = float(input_array[71])
    machine.phi12 = float(input_array[73])
    machine.b0 = float(input_array[75])
    machine.bdot = float(input_array[77])
    machine.mean_orbit_rad = float(input_array[79])
    machine.bending_rad = float(input_array[81])
    machine.trans_gamma = float(input_array[83])
    machine.e_rest = float(input_array[85])
    machine.q = float(input_array[87])
    machine.self_field_flag = bool(int(input_array[91]))
    machine.g_coupling = float(input_array[93])
    machine.zwall_over_n = float(input_array[95])
    machine.pickup_sensitivity = float(input_array[97])
    return machine



