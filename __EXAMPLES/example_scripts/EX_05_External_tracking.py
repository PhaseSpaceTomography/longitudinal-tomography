import numpy as np
import sys
import os
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
    
    with open(input_file) as f:
        read = f.readlines()

    # Splitting to parameters and meassured profile data
    raw_parameters = read[:len_input_param]
    raw_data = np.array(read[len_input_param:], dtype=float)
    del(read)

    # Setting output directory
    output_dir_idx = 14
    raw_parameters[output_dir_idx] = f'{output_dir}'

    param = Parameters()
    param.parse_from_txt(raw_parameters)
    param.fill()

    rf1vs = (param.vrf1 + param.vrf1dot * param.time_at_turn) * param.q
    rf2vs = (param.vrf2 + param.vrf2dot * param.time_at_turn) * param.q

    # Call tracking routine



    raise NotImplementedError('Example currently under construction.')

    
if __name__ == '__main__':
    main()
