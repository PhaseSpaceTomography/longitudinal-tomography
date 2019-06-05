import os
import sys
import subprocess
import ctypes

path = os.path.realpath(__file__)
basepath = os.sep.join(path.split(os.sep)[:-1]) + "/cpp_files/"

# print(basepath)
# print(os.listdir(basepath + "/cpp_files"))

# TODO: Add arguments?
# TODO: Add compilation for windows?

c_flags = ["-std=c++11", "-fopenmp", "-shared",  "-fPIC",
           "-O3", "-march=native", "-ffast-math"]

cpp_files = [
    os.path.join(basepath, "longtrack.cpp"),
    os.path.join(basepath, "map_weights.cpp")
]

libname = os.path.join(basepath, 'tomolib.so')

compiler = "g++"

if __name__ == '__main__':
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
        print("\nCompilation succeeded!")
    except Exception as e:
        print("\nCompilation failed.")
        print(e)
