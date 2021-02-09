import os
import sys
import subprocess
import ctypes
import argparse


def main():
    path = os.path.realpath(__file__)
    base_path = os.sep.join(path.split(os.sep)[:-2])
    cpp_dir_path = base_path + '/longitudinal_tomography/cpp_routines/'

    cpp_files = [os.path.join(cpp_dir_path, 'reconstruct.cpp'),
                 os.path.join(cpp_dir_path, 'kick_and_drift.cpp')]

    # Sets flags
    c_flags = ['-std=c++11', '-shared', '-O3']
    c_flags += ['-fopenmp', '-march=native', '-ffast-math']

    compiler = 'g++'

    # Setting system spescific parameters
    if 'posix' in os.name:
        c_flags += ['-fPIC']
        libname = os.path.join(cpp_dir_path, 'tomolib.so')
    elif 'win' in sys.platform:
        # TODO: Find out if this works
        libname = os.path.join(cpp_dir_path, 'tomolib.dll')
    else:
        print('YOU ARE NOT USING A WINDOWS'
              'OR LINUX OPERATING SYSTEM. ABORTING...')
        sys.exit(-1)

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

if __name__ == '__main__':
    main()
