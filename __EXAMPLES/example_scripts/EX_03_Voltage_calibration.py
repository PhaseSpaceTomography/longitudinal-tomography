import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from utils.exs_tools import show, make_or_clear_dir
sys.path.append('../../tomo')      # Hack
from main import main as tomo_main # Hack
from parameters import Parameters  # Hack

BASE_PATH = os.path.dirname(
                os.path.realpath(__file__)).split('/')[:-1]
BASE_PATH = '/'.join(BASE_PATH)

INPUT_FILE_DIR = '/'.join([BASE_PATH] + ['input_files'])

def main():
    input_file = f'{INPUT_FILE_DIR}/flatTopINDIVRotate3.dat'
    output_dir = f'{BASE_PATH}/tmp'

    make_or_clear_dir(output_dir)

    len_input_param = 98

    # Reading input file to memory
    with open(input_file) as f:
        read = f.readlines()

    # Splitting to parameters and meassured profile data
    raw_parameters = read[:len_input_param]
    raw_data = np.array(read[len_input_param:], dtype=float)
    del(read)

    # Setting output directory
    output_dir_idx = 14
    raw_parameters[output_dir_idx] = f'{output_dir}\r\n'

    nsteps = 20
    rfv_raw_data_idx = 61
    default_voltage = float(raw_parameters[rfv_raw_data_idx])
    start_voltage = default_voltage - 500
    stop_voltage = default_voltage + 500
    last_differences = []
    input_volts = []
    for voltage in np.linspace(start_voltage, stop_voltage, nsteps):
        raw_parameters[rfv_raw_data_idx] = f'{voltage}\r\n'
        param = Parameters()
        param.parse_from_txt(raw_parameters)
        _, diff, _ = tomo_main(param, raw_data)
        last_differences.append(diff[-1])
        input_volts.append(voltage)

    show(input_volts, last_differences)

def show(input_volts, output_discr):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.set_title('Voltage calibration')
    ax.scatter(input_volts, output_discr)
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Discrepancy')

    ymin = min(output_discr)
    ymax = max(output_discr)
    temp = (ymax - ymin) * 0.05
    ymin -= temp
    ymax += temp

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_ylim([ymin, ymax])
    plt.show()

if __name__ == '__main__':
    main()
