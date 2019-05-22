import matplotlib.pyplot as plt
from tomo_action.analyze import Analyze
from tomo_action.system_handling import SysHandling

TMP_DIR = r"tmp"
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

    SysHandling.dir_exists(TMP_DIR)
    SysHandling.clear_dir(TMP_DIR)
    sysh = SysHandling(INPUT_NAMES, TMP_DIR, RESOURCES_DIR, PY_MAIN_PATH)

    for i in range(start_file, end_file):

        plt.figure(INPUT_NAMES[i])
        plt.clf()

        if not load_from_file:
            in_path = TMP_DIR
            sysh.run_programs(i, python=True, fortran=True)
            (py_image,
             f_image) = SysHandling.get_pics_from_file(in_path)
        else:
            print("\n!! loading from files !!")
            print("current input: " + INPUT_NAMES[i])
            in_path = create_out_dir_path(INPUT_NAMES[i])
            (py_image,
             f_image) = SysHandling.get_pics_from_file(in_path)

        if show_picture:
            Analyze.show_images(py_image, f_image, plt)

        if analyze:
            do_analyze(in_path, py_image, f_image, plot=plt)

        if show_picture or analyze:
            plt.show()

        if not load_from_file:
            sysh.move_to_resource_dir(i)


# Asserts that the path to output dir is created correctly
def create_out_dir_path(input_name):
    if RESOURCES_DIR[-1] != "/":
        return RESOURCES_DIR + "/" + input_name
    else:
        return RESOURCES_DIR + input_name


def do_analyze(path, py_image, f_image, plot):
    plt.subplot(224)
    plt.axis('off')
    f_diff, py_diff = SysHandling.find_difference_files(path)
    Analyze.show_difference_to_original(f_diff, py_diff, plot)

    pf_diff = Analyze.compare_phase_space(py_image, f_image)
    Analyze.show_difference_to_fortran(pf_diff, plot)

    Analyze.compare_profiles(path, py_image, f_image, plot)


if __name__ == '__main__':
    main(load_from_file=False,
         show_picture=True,
         analyze=True,
         start_file=0,
         end_file=1)
