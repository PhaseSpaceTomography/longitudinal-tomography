import logging
import matplotlib.pyplot as plt
from analyze import Analyze
from system_handling import SysHandling

logging.basicConfig(level=logging.INFO)

WORKING_DIR = r"tmp"
RESOURCES_DIR = r"test_resources"
OUTPUT_DIR = r"saved_output"
PY_MAIN_PATH = r"../tomo/Main_testing.py"

INPUT_NAMES = [                     # Nbr
    "C500MidPhaseNoise",            # 0
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
    "noiseStructure2"               # 11
]


def main(load_from_file=False, show_picture=True,
         analyze=True, start_file=0, end_file=len(INPUT_NAMES)):

    SysHandling.clear_dir(WORKING_DIR)
    sysh = SysHandling(INPUT_NAMES, WORKING_DIR, RESOURCES_DIR, PY_MAIN_PATH)

    for i in range(start_file, end_file):

        plt.figure(INPUT_NAMES[i])
        plt.clf()

        if not load_from_file:
            sysh.run_programs(i, python=True, fortran=True)
            (py_picture,
             ftr_picture) = SysHandling.get_pics_from_file(WORKING_DIR)
        else:
            print("\n!! loading from files !!")
            print("current input: " + INPUT_NAMES[i])
            inpath = create_out_dir_path(INPUT_NAMES[i])
            (py_picture,
             ftr_picture) = SysHandling.get_pics_from_file(inpath)

        if show_picture:
            Analyze.show_images(py_picture, ftr_picture, plt)

        if analyze:
            if load_from_file:
                Analyze.analyze_difference(inpath, plot=plt)
                Analyze.compare_profiles(inpath, plot=plt)
            else:
                Analyze.analyze_difference(WORKING_DIR, plot=plt)
                Analyze.compare_profiles(WORKING_DIR, plot=plt)

        if show_picture or analyze:
            plt.show()

        if not load_from_file:
            sysh.move_to_resource_dir(i)


def create_out_dir_path(input_name):
    if RESOURCES_DIR[-1] != "/":
        return RESOURCES_DIR + "/" + input_name
    else:
        return RESOURCES_DIR + input_name


if __name__ == '__main__':
    main(load_from_file=True,
         show_picture=True,
         analyze=True,
         start_file=0,
         end_file=1)
