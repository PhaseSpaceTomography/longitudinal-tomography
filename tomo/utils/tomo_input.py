import numpy as np
import logging as log
import os
import sys
from machine import Machine
from .exceptions import InputError
from .assertions import assert_inrange

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
    return np.array(read)


# Read machine parameters via stdin.
# Here the measured data must be pipelined in the same file as
# the machine parameters.
def _get_input_stdin():
    read = []
    finished = False
    piped_raw_data = False
    
    line_num = 0
    ndata_points = PARAMETER_LENGTH

    while line_num < ndata_points:
        read.append(sys.stdin.readline())
        if line_num == RAW_DATA_FILE_IDX:
            if 'pipe' in read[-1]:
                piped_raw_data = True
        if piped_raw_data:
            if line_num == 16:
                nframes = int(read[-1])
            if line_num == 20:
                nbins = int(read[-1])
                ndata_points += nframes * nbins
        if line_num == ndata_points:
            finished = True
        line_num += 1
    return np.array(read)


# Splits the read input data to machine parameters and raw data.
# If the raw data is not already read from the input file, the
#  data will be found in the file given by the parameter file.
def _split_input(read_input):
    nframes_idx = 16
    nbins_idx = 20
    ndata = 0
    read_parameters = None
    read_data = None
        
    try:
        read_parameters = read_input[:PARAMETER_LENGTH]
        ndata = (int(read_parameters[nbins_idx])
                 * int(read_parameters[nframes_idx]))
        for i in range(PARAMETER_LENGTH):
            read_parameters[i] = read_parameters[i].strip('\r\n')
    except:
        err_msg = 'Something went wrong while accessing machine parameters.'
        raise InputError(err_msg)

    if read_parameters[RAW_DATA_FILE_IDX] == 'pipe':
        try:
            read_data = np.array(read_input[PARAMETER_LENGTH:], dtype=float)
        except:
            err_msg = 'Pipelined raw-data could not be casted to float.'
            raise InputError(err_msg)
    else:
        try:
            read_data = np.genfromtxt(read_parameters[RAW_DATA_FILE_IDX],
                                      dtype=float)
        except FileNotFoundError:
            err_msg = f'The given file path for the raw-data:\n'\
                      f'{read_parameters[RAW_DATA_FILE_IDX]}\n'\
                      f'Could not be found'
            raise FileNotFoundError(err_msg)
        except Exception:
            err_msg = 'Something went wrong while loading raw_data.'
            raise Exception(err_msg)
            

    if not len(read_data) == ndata:
        raise InputError(f'Wrong amount of datapoints loaded.\n'
                         f'Expected: {ndata}\n'
                         f'Loaded:   {len(read_data)}')


    return read_parameters, read_data


# Function to convert from array containing the lines in an input file
#  to a partially filled machine object.
# The array must contain a direct read from an input file.
# Some variables are subtracted by one.
#  This is in order to convert from Fortran to C style indexing.
#  Fortran counts (by defalut) from 1.
def input_to_machine(input_array):
    if len(input_array) != PARAMETER_LENGTH:
        raise InputError

    for i in range(len(input_array)):
            input_array[i] = input_array[i].strip('\r\n')

    machine = Machine()
    machine.rawdata_file        = input_array[12]
    machine.output_dir          = input_array[14]
    machine.framecount          = int(input_array[16])
    machine.frame_skipcount     = int(input_array[18])
    machine.framelength         = int(input_array[20])
    machine.dtbin               = float(input_array[22])
    machine.dturns              = int(input_array[24])
    machine.preskip_length      = int(input_array[26])
    machine.postskip_length     = int(input_array[28])
    machine.imin_skip           = int(input_array[31])
    machine.imax_skip           = int(input_array[34])
    machine.rebin               = int(input_array[36])
    machine._xat0               = float(input_array[39])
    machine.demax               = float(input_array[41])
    machine.filmstart           = int(input_array[43]) -1
    machine.filmstop            = int(input_array[45])
    machine.filmstep            = int(input_array[47])
    machine.niter               = int(input_array[49])
    machine.snpt                = int(input_array[51])
    machine.full_pp_flag        = bool(int(input_array[53]))
    machine.beam_ref_frame      = int(input_array[55]) - 1
    machine.machine_ref_frame   = int(input_array[57]) - 1
    machine.vrf1                = float(input_array[61])
    machine.vrf1dot             = float(input_array[63])
    machine.vrf2                = float(input_array[65])
    machine.vrf2dot             = float(input_array[67])
    machine.h_num               = float(input_array[69])
    machine.h_ratio             = float(input_array[71])
    machine.phi12               = float(input_array[73])
    machine.b0                  = float(input_array[75])
    machine.bdot                = float(input_array[77])
    machine.mean_orbit_rad      = float(input_array[79])
    machine.bending_rad         = float(input_array[81])
    machine.trans_gamma         = float(input_array[83])
    machine.e_rest              = float(input_array[85])
    machine.q                   = float(input_array[87])
    machine.self_field_flag     = bool(int(input_array[91]))
    machine.g_coupling          = float(input_array[93])
    machine.zwall_over_n        = float(input_array[95])
    machine.pickup_sensitivity  = float(input_array[97])
    return machine

# Convert from one-dimentional list of raw data to waterfall.
# Works on a copy of the raw data
def raw_data_to_waterfall(machine, raw_data):
    waterfall = np.copy(raw_data)
    waterfall = raw_data.reshape((machine.nprofiles, machine.framelength))

    if machine.postskip_length > 0:
        waterfall = waterfall[:, machine.preskip_length:
                                -machine.postskip_length]
    else:
        waterfall = waterfall[:, machine.preskip_length:]

    return waterfall


# Original function for subtracting baseline of raw data input profiles.
# Finds the baseline from the first 5% (by default)
#  of the beam reference profile.
def _calc_baseline_ftn(waterfall, ref_prof, percent=0.05):
    assert_inrange(percent, 'percent', 0.0, 1.0, InputError,
                   'The chosen percent of raw_data '
                   'to create baseline from is not valid')

    nbins = len(waterfall[ref_prof])
    iend = int(percent * nbins) 

    return np.sum(waterfall[ref_prof, :iend]) / np.floor(percent * nbins)


def rebin(waterfall, rbn):
    data = np.copy(waterfall)

    # Check that there is enough data to for the given rebin factor.

    if data.shape[1] % rbn == 0:
        rebinned = _rebin_even(data, rbn)
    else:
        rebinned = _rebin_odd(data, rbn)

    return rebinned

# Rebins an 2d array given a rebin factor (rbn).
# The given array MUST have a length equal to an even number.
def _rebin_even(data, rbn):
    if data.shape[1] % rbn != 0:
        raise AssertionError('Length of input data must be an even number.')

    ans = np.copy(data)
    
    nprofs = data.shape[0]
    nbins = data.shape[1]

    new_nbins = int(nbins / rbn)
    all_bins = new_nbins * nprofs
    
    ans = ans.reshape((all_bins, rbn))
    ans = np.sum(ans, axis=1)
    ans = ans.reshape((nprofs, new_nbins))

    return ans


# Rebins an 2d array given a rebin factor (rbn).
# The given array MUST have vector length equal to an odd number.
def _rebin_odd(data, rbn):
    nprofs = data.shape[0]
    nbins = data.shape[1]
    ans = np.zeros((nprofs, int(nbins / rbn) + 1))
    ans[:,:-1] = _rebin_even(data[:,:-1], rbn)
    ans[:,-1] = _rebin_last(data, rbn)[:, 0]
    return ans


# Rebins last indices of an 2d array given a rebin factor (rbn).
# Needed for the rebinning of odd arrays.
def _rebin_last(data, rbn):
    nprofs = data.shape[0]
    nbins = data.shape[1]
    new_nbins = int(nbins / rbn)

    i0 = (new_nbins - 1) * rbn
    ans = np.copy(data[:,i0:])
    ans = np.sum(ans, axis=1)
    ans[:] *= rbn / (nbins - (new_nbins - 1) * rbn)
    ans = ans.reshape((nprofs, 1))
    return ans
