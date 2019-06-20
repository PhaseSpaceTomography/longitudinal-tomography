import matplotlib.pyplot as plt
import argparse
from tomo_action.analyze import Analyze
from tomo_action.system_handling import SysHandling

TMP_DIR = r"tmp"
RESOURCES_DIR = r"test_resources"
OUTPUT_DIR = r"saved_output"
PY_MAIN_PATH = r"../tomo/main_testing.py"

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


def main(load_image, analyze, start_file, end_file, custom_input):
    custom_in_flag = False
    if len(custom_input) > 0:
        input_list = custom_input
        resources_dir = ''
        custom_in_flag = True
        start_file = 0
        end_file = len(custom_input)
    else:
        input_list = INPUT_NAMES
        resources_dir = RESOURCES_DIR

    SysHandling.dir_exists(TMP_DIR)
    SysHandling.clear_dir(TMP_DIR)
    sysh = SysHandling(input_list, TMP_DIR, resources_dir, PY_MAIN_PATH)

    for i in range(start_file, end_file):

        plt.figure(input_list[i])
        plt.clf()

        if not load_image:
            in_path = TMP_DIR

            time_py, time_f = sysh.run_programs(i, python=True, fortran=True)
            (py_image,
             f_image) = SysHandling.get_pics_from_file(in_path)
        else:
            time_py = float('nan')
            time_f = float('nan')
            print("\n!! loading from files !!")
            print("current input: " + input_list[i])
            in_path = create_out_dir_path(input_list[i])
            (py_image,
             f_image) = SysHandling.get_pics_from_file(in_path)

        if analyze:
            Analyze.show_images(py_image, f_image, plt)
            do_analyze(in_path, py_image, f_image, plot=plt)
            print("Execution time python: " + str(time_py))
            print("Execution time fortran: " + str(time_f))
            plt.show()

        if not load_image and not custom_in_flag:
            sysh.move_to_resource_dir(i)


# Asserts that the path to output dir is created correctly
def create_out_dir_path(input_name):
    if RESOURCES_DIR[-1] != "/":
        return RESOURCES_DIR + "/" + input_name
    else:
        return RESOURCES_DIR + input_name


def do_analyze(path, py_image, f_image, plot):
    plt.subplot(325)
    plt.axis('off')
    f_diff, py_diff = SysHandling.find_difference_files(path)
    Analyze.show_difference_to_original(f_diff, py_diff, plot)

    pf_diff = Analyze.compare_phase_space(py_image, f_image)
    Analyze.write_difference_py_ftr(pf_diff, plot)
    Analyze.plot_difference_py_ftr(py_image, f_image, plot)

    Analyze.compare_profiles(path, py_image, f_image, plot)


def assert_args(args):
    if args.start < 0 or args.start >= args.end or args.end > len(INPUT_NAMES):
        raise SystemExit("\nBad start/end input\n"
                         "start and end must be between "
                         "0 and " + str(len(INPUT_NAMES))
                         + "\nProgram stopped...")


def create_parser():
    input_help_str = ""
    for i in range(len(INPUT_NAMES)):
        input_help_str += f"{i}:\t{INPUT_NAMES[i]}\n"

    parser = argparse.ArgumentParser(description='Run tomo_action main to '
                                                 'easily run the tomography '
                                                 'application with test '
                                                 'inputs\n\n'
                                                 + input_help_str)

    parser.add_argument('-a', '--analyse',
                        default=False,
                        type=bool,
                        nargs='?',
                        const=True,
                        help='Show phase space plot and compare'
                             'python output with Fortran')

    parser.add_argument('-l', '--load',
                        default=False,
                        type=bool,
                        nargs='?',
                        const=True,
                        help='Load already reconstructed image')

    parser.add_argument('-start', '--start',
                        default=0,
                        type=int,
                        help="First input file")

    parser.add_argument('-end', '--end',
                        default=len(INPUT_NAMES),
                        type=int,
                        help="Last input file")

    parser.add_argument('-ci', '--custom_input',
                        default=[],
                        nargs='+')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    assert_args(args)

    main(load_image=args.load,
         analyze=args.analyse,
         start_file=args.start,
         end_file=args.end,
         custom_input=args.custom_input)
