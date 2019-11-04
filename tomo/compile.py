import os
import sys
import subprocess
import ctypes
import argparse


# TODO: Add arguments?
# TODO: Add compilation for windows?

def main(gpu_flag):
    path = os.path.realpath(__file__)
    basepath = os.sep.join(path.split(os.sep)[:-1]) + '/cpp_routines/'

    cpp_files = [os.path.join(basepath, 'reconstruct.cpp'),
                 os.path.join(basepath, 'kick_and_drift.cpp')]

    # Sets flags
    c_flags = ['-std=c++11', '-shared', '-O3']
    if gpu_flag:
        # Now asumes device type: nvidia. Add argument to alter?
        # c_flags += ['-Minfo=accel'] <- Verbose mode
        c_flags += ['-acc', '-ta=nvidia']
    else:
        c_flags += ['-fopenmp', '-march=native', '-ffast-math']

    # Choose compiler:
    if gpu_flag:
        compiler = 'pgc++'
    else:
        compiler = 'g++'

    # Setting system spescific parameters
    if 'posix' in os.name:
        c_flags += ['-fPIC']
        libname = os.path.join(basepath, 'tomolib.so')
    elif 'win' in sys.platform:
        # TODO: Find out if this works
        libname = os.path.join(basepath, 'tomolib.dll')
    else:
        print('YOU ARE NOT USING A WINDOWS'
              'OR LINUX OPERATING SYSTEM. ABORTING...')
        sys.exit(-1)

    # Write compilation info to user
    if gpu_flag:
        print('\nCompilation mode: GPU')
    else:
        print('\nCompilation mode: CPU')

    print('C++ Compiler: ', compiler)
    print('Compiler flags: ', ' '.join(c_flags))
    subprocess.call([compiler, '--version'])

    try:
        os.remove(libname)
    except OSError as e:
        pass

    command = [compiler] + c_flags + ['-o', libname] + cpp_files
    subprocess.call(command)

    try:
        lib = ctypes.CDLL(libname)
        print('\nCompilation succeeded!')
    except Exception as e:
        print('\nCompilation failed.')
        # print(f'Error: {e}')


def _get_parser():
    parser = argparse.ArgumentParser(
                description='Compile c++ functions for tomography program.')

    parser.add_argument('-gpu', '--GPU',
                        default=False,
                        type=bool,
                        nargs='?',
                        const=True,
                        help='Compile for gpu')
    return parser

if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()
    main(args.GPU)
