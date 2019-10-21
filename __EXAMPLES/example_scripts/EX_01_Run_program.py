# For running program
import os
# For presentation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


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
    show(output_dir)


def show(output_dir):
    image, profile, diff = get_out_files(output_dir)
    rec_prof = np.sum(image, axis=1)


    gs = gridspec.GridSpec(2, 2)
    
    fig = plt.figure(figsize=(15, 6))
    
    img = fig.add_subplot(gs[:, 1])
    profs = fig.add_subplot(gs[0, 0])
    convg = fig.add_subplot(gs[1, 0])

    # Showing phase-space
    img.set_title('Reconstructed phase-space')
    hot_im = img.imshow(image.T, origin='lower',
                        interpolation='nearest', cmap='hot')
    divider = make_axes_locatable(img)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(hot_im, cax=cax, format='%.0e')
    

    # plotting measured and recreated profile
    profs.set_title('Reconstructed vs measured profile')
    profs.plot(profile, label='measured')
    profs.plot(rec_prof, label='reconstructed')
    profs.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    profs.legend()
    
    # plotting convergence
    convg.set_title('Distrepancy for each iteration of reconstruction')
    convg.plot(diff)
    convg.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.show()


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


def make_or_clear_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        files = os.listdir(dir_name)
        for file in files:
            if os.path.isfile(f'{dir_name}/{file}'):
                os.remove(f'{dir_name}/{file}')


if __name__ == '__main__':
    main()