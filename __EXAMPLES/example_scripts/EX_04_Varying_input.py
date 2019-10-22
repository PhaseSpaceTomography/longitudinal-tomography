import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import time as tm
from utils.exs_tools import show, make_or_clear_dir
sys.path.append('../../tomo')      # Hack
from main import main as tomo_main # Hack
from parameters import Parameters  # Hack

BASE_PATH = os.path.dirname(
                os.path.realpath(__file__)).split('/')[:-1]
BASE_PATH = '/'.join(BASE_PATH)

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
    del(read)

    # Setting output directory
    output_dir_idx = 14
    raw_parameters[output_dir_idx] = f'{output_dir}'

    # Function for varying python input
    rebin_idx = 36
    snpt_idx = 51

    snpt_py_vals = python_varying_input(raw_parameters, raw_data,
                                        snpt_idx, (1, 4))
    rebin_py_vals = python_varying_input(raw_parameters, raw_data,
                                         rebin_idx, (6, 8))

    plot(snpt_py_vals, 'snpt')
    plot(rebin_py_vals, 'rebin')


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

    rebin = 0
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
            err_msg = f'Something went wring when running with '\
                      f'input parameter of the value: {in_param}.'
            print(err_msg)
            break
    return vals

def plot(py_values, tag):
    parameter_values = py_values[:, 0].astype(int).astype(str)
    x = py_values[:, 2] # Time
    y = py_values[:, 1] # Discrepancy
    y = y * 1000

    margin = max(y)*0.005 

    fig, ax = plt.subplots()

    for i, par_val in enumerate(parameter_values):
        ax.scatter(x[i], y[i], color='blue')
        ax.text(x[i]+margin, y[i]+margin, par_val, fontsize=9)
    
    ax.set_title(f'Time vs discrepancy - varying the {tag} parameter ')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Discrepancy (x1000)')

    plt.show()

if __name__ == '__main__':
    main()
