import os
import numpy as np
from utils.exs_tools import show, make_or_clear_dir
                                               # Idx
INPUT_NAMES = ["C500MidPhaseNoise",            # 0
               "C550MidPhaseNoise",            # 1
               "flatTopINDIV8thOrder",         # 2
               "flatTopINDIVRotate",           # 3
               "flatTopINDIVRotate2",          # 4
               "flatTopINDIVRotate3",          # 5
               "flatTopINDIVRotateCalibPS",    # 6
               "INDIVShavingC325",             # 7
               "MidINDIVNoiseC350",            # 8
               "MidINDIVNoiseC350-2",          # 9
               "noiseStructure1",              # 10
               "noiseStructure2"]              # 11
              
BASE_PATH = os.path.os.path.dirname(
                os.path.realpath(__file__)).split('/')[:-1]
BASE_PATH = '/'.join(BASE_PATH)

INPUT_FILE_DIR = '/'.join([BASE_PATH] + ['input_files'])

TOMO_PATH = os.path.os.path.dirname(
                os.path.realpath(__file__)).split('/')[:-2]
TOMO_PATH = '/'.join(TOMO_PATH + ['tomo']) 


def main(input_name_idx=7):
    make_or_clear_dir(f'{BASE_PATH}/tmp')
    file_path = f'{INPUT_FILE_DIR}/{INPUT_NAMES[input_name_idx]}.dat'
    output_dir = f'{BASE_PATH}/tmp'

    # Before the program can be run, the necessary C++ library must be compiled.
    cpp_library = f'{TOMO_PATH}/cpp_routines/tomolib.so' 
    if not os.path.isfile(cpp_library):
        compile_script = f'{TOMO_PATH}/compile.py'
        command = f'python {compile_script}'
        os.system(command)

    # Possible commands for running tomography:
    # ---------------------------------
    #  1. Read file via stdin*
    #       python .../main.py < .../input_file.dat
    #
    #  2. Give input file by file path*
    #       python .../main.py .../input_file.dat
    #
    #  3. Give input file and output directory by path
    #       python .../main.py .../input_file.dat .../out/
    # 
    # *Saves output files to directory spescified in input file.        

    command = f'python {TOMO_PATH}/main.py {file_path} {output_dir}'
    os.system(command)
    
    # Retrieving data from files
    image, profile, diff = get_out_files(output_dir)
    show(image, diff, profile)

def get_out_files(output_dir):
    out_files = os.listdir(output_dir)
    
    measured_profile = None
    phase_space_image = None
    discrepancies = None

    # The program produces the following output files:
    # -----------------------------------------------------------------------
    # dXXX.data         - discrepancies for each iteration of reconstruction
    # imageXXX.data     - recreated phase-space
    # profileXXX.data   - measured profile to be recreated.
    # (vself.data)      - calculated self fields in profiles.
    #                     Appears only if self field flag is 1 in input file.

    for f in out_files:
        if f.startswith('profile'):
            measured_profile = np.genfromtxt(f'{output_dir}/{f}')
        elif f.startswith('d'):
            discrepancies = np.genfromtxt(f'{output_dir}/{f}')
        elif f.startswith('image'):
            phase_space_image = np.genfromtxt(f'{output_dir}/{f}')

    # Deleting column of indices 
    #  showing the iteration where the discrepancy is calculated.   
    discrepancies = np.delete(discrepancies, 0, axis=1)

    # Reshaping phase-space image from 1D to 2D array.
    # The image is always quadratic.
    side = int(np.sqrt(len(phase_space_image)))
    phase_space_image = phase_space_image.reshape((side, side))

    return phase_space_image, measured_profile, discrepancies

if __name__ == '__main__':
    main()