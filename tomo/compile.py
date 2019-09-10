import os
import sys
import subprocess
import ctypes

path = os.path.realpath(__file__)
basepath = os.sep.join(path.split(os.sep)[:-1]) + '/cpp_routines/'

# TODO: Add arguments?
# TODO: Add compilation for windows?

c_flags = ['-std=c++11', '-fopenmp', '-shared',
           '-O3', '-march=native', '-ffast-math']

cpp_files = [
    os.path.join(basepath, 'drift.cpp'),
    os.path.join(basepath, 'kick.cpp'),
    os.path.join(basepath, 'tomo_routines.cpp')]

compiler = 'g++'

if __name__ == '__main__':
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
