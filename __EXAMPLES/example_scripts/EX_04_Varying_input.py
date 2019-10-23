import matplotlib.pyplot as plt
import os
import sys
import subprocess as sub
import numpy as np
import time as tm
from utils.exs_tools import show, make_or_clear_dir
sys.path.append('../../tomo')      # Hack
from main import main as tomo_main # Hack
from parameters import Parameters  # Hack

BASE_PATH = os.path.dirname(
                os.path.realpath(__file__)).split('/')[:-1]
BASE_PATH = '/'.join(BASE_PATH)

FORTRAN_EXE = f'{BASE_PATH}/fortran/tomo_vo.intelmp'

INPUT_FILE_DIR = '/'.join([BASE_PATH] + ['input_files'])


def main():
    len_input_param = 98
    input_file = f'{INPUT_FILE_DIR}/C500MidPhaseNoise.dat'
    output_dir = f'{BASE_PATH}/tmp'

    make_or_clear_dir(output_dir)

    # Reading input file to memory
    with open(input_file) as f:
        read = f.readlines()

    # Splitting to parameters and meassured profile data
    raw_parameters = read[:len_input_param]
    raw_data = np.array(read[len_input_param:], dtype=float)

    # Setting output directory
    output_dir_idx = 14
    raw_parameters[output_dir_idx] = f'{output_dir}'

    rebin_idx = 36
    snpt_idx = 51
    snpt_py = python_varying_input(raw_parameters, raw_data,
                                        snpt_idx)
    rebin_py = python_varying_input(raw_parameters, raw_data,
                                         rebin_idx)

    snpt_ftr = fortran_varying_input(read, output_dir,
                                          snpt_idx)
    rebin_ftr = fortran_varying_input(read, output_dir,
                                           rebin_idx)

    plot(snpt_py, snpt_ftr, 'snpt')
    plot(rebin_py, rebin_ftr, 'rebin')


def python_varying_input(raw_param, raw_data, input_idx, par_range=(1, 10)):
    raw_param = raw_param.copy()
    inputs = np.arange(par_range[0], par_range[1])
    
    # Initiating to -1 to later see which have been successfully run.
    # First collumn of array is the input parameter
    # Second collumn of array is the output discrepancy of the last iteration
    #  of the reconstruction.
    # Third collumn is the run time of the full reconstruction
    vals = np.zeros((len(inputs), 3)) - 1
    vals[:,0] = inputs

    failed_runs = []
    for i, in_param in enumerate(inputs):
        raw_param[input_idx] = str(in_param)
        param = Parameters()
        param.parse_from_txt(raw_param)
        try:
            t0 = tm.perf_counter()
            _, diff, _ = tomo_main(param, raw_data)
            t1 = tm.perf_counter()
            vals[i, 1] = diff[-1]
            vals[i, 2] = t1 - t0 
        except:
            failed_runs.append(i)

    vals = np.delete(vals, failed_runs, axis=0)
    return vals

def fortran_varying_input(input_file, output_dir,
                          input_idx, par_range=(1, 10)):
    input_file = input_file.copy()

    inputs = np.arange(par_range[0], par_range[1])
    vals = np.zeros((len(inputs), 3)) - 1
    vals[:,0] = inputs

    # Path as seen from Fortran executable
    # NB: The output path can have the max length of 60 characters
    #  for the Fortran program.
    #  If this number is exceeded, the output will be
    #  saved to the wrong location.
    output_idx = 14
    input_file[output_idx] = '../tmp/\r\n'

    failed_runs = []
    for i, in_param in enumerate(inputs):
        
        make_or_clear_dir(output_dir)
        
        input_file[input_idx] = f'{in_param}\r\n'
        
        # Save new Fortran input file with updated parameters
        ftr_input_path = make_new_ftr_input(input_file, output_dir)
        
        # Run Fortran exe
        with open(ftr_input_path) as f_input:
            t0 = tm.perf_counter()
            failed = sub.call([FORTRAN_EXE], stdin=f_input)
            t1 = tm.perf_counter()

        if not failed:
            diff = find_fortran_discrepancies(output_dir)
            vals[i, 1] = diff[-1]
            vals[i, 2] = t1 - t0
        else:
            failed_runs.append(i)

    vals = np.delete(vals, failed_runs, axis=0)

    return vals


# Save input file for fortran exe
def make_new_ftr_input(input_file, output_dir):
    ftr_input_path = f'{output_dir}/input.dat' 
    with open(ftr_input_path, 'w') as f:
        for line in input_file:
            f.write(line)
    return ftr_input_path

def find_fortran_discrepancies(output_dir):
    out_files = os.listdir(output_dir)

    discrepancies = None
    for f in out_files:
        if f.startswith('d') and f.endswith('.data'):
            discrepancies = np.genfromtxt(f'{output_dir}/{f}')
    discrepancies = np.delete(discrepancies, 0, axis=1)

    return discrepancies


def plot(py_values, ftr_values, tag):
    py_vals = py_values[:, 0].astype(int).astype(str)
    py_x = py_values[:, 2] # Time
    py_y = py_values[:, 1] # Discrepancy
    py_y = py_y * 1000
    margin = max(py_y)*0.005

    ftr_vals = ftr_values[:, 0].astype(int).astype(str)
    ftr_x = ftr_values[:, 2] # Time
    ftr_y = ftr_values[:, 1] # Discrepancy
    ftr_y = ftr_y * 1000 

    fig, ax = plt.subplots()

    ax.scatter(py_x, py_y, color='orange', label='python')
    for i, par_val in enumerate(py_vals):
        ax.text(py_x[i]+margin, py_y[i]+margin, par_val, fontsize=9)

    ax.scatter(ftr_x, ftr_y, color='green', label='fortran')
    for i, par_val in enumerate(ftr_vals):
        ax.text(ftr_x[i]+margin, ftr_y[i]+margin, par_val, fontsize=9)
    
    ax.legend()
    ax.set_title(f'Time vs discrepancy - varying the {tag} parameter ')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Discrepancy (x1000)')

    plt.show()

if __name__ == '__main__':
    main()
