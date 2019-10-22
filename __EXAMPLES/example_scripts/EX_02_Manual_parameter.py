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

TOMO_PATH = os.path.dirname(
                os.path.realpath(__file__)).split('/')[:-2]
TOMO_PATH = '/'.join(TOMO_PATH + ['tomo']) 


def main():
    data_path = f'{INPUT_FILE_DIR}/C500MidPhaseNoise_data.dat'
    output_dir = f'{BASE_PATH}/tmp/' 

    make_or_clear_dir(output_dir)

    # Manually giving value to each field in object
    param = fill_up_parameter(data_path, output_dir)

    # tomo main takes a partially filled parameter.
    image, diff, profile = tomo_main(param)

    show(image, diff, profile)


# Filling parameter manually
# Example retrieved from .../input_files/C500MidPhaseNoise.dat
def fill_up_parameter(data_path, output_dir):
    p = Parameters()
    p.rawdata_file          = data_path
    p.output_dir            = output_dir
    p.framecount            = 100
    p.frame_skipcount       = 0
    p.framelength           = 1000
    p.dtbin                 = 4.999999999999999E-10
    p.dturns                = 12
    p.preskip_length        = 130
    p.postskip_length       = 50
    p.imin_skip             = 0
    p.imax_skip             = 0
    p.rebin                 = 4
    p.xat0                  = 352.00000000000006
    p.demax                 = -1.E6
    p.filmstart             = 1
    p.filmstop              = 1
    p.filmstep              = 1
    p.num_iter              = 20
    p.snpt                  = 4
    p.full_pp_flag          = False
    p.beam_ref_frame        = 1
    p.machine_ref_frame     = 1
    p.vrf1                  = 7945.403672852664
    p.vrf1dot               = 0.0
    p.vrf2                  = -0
    p.vrf2dot               = 0.0
    p.h_num                 = 1
    p.h_ratio               = 2
    p.phi12                 = 0.3116495273194016
    p.b0                    = 0.38449
    p.bdot                  = 1.882500000000023
    p.mean_orbit_rad        = 25.0
    p.bending_rad           = 8.239
    p.trans_gamma           = 4.1
    p.e_rest                = 0.93827231E9
    p.q                     = 1
    p.self_field_flag       = 0
    p.g_coupling            = 0.0
    p.zwall_over_n          = 0.0
    p.pickup_sensitivity    = 0.36
    return p


if __name__ == '__main__':
    main()
