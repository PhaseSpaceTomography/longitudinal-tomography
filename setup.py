from setuptools import setup
from os import path
from pybind11.setup_helpers import Pybind11Extension

HERE = path.split(path.abspath(__file__))[0]

cpp_routines = Pybind11Extension('tomo.cpp_routines.libtomo',
                                 sources=['tomo/cpp_routines/libtomo.cpp',
                                          'tomo/cpp_routines/reconstruct.cpp',
                                          'tomo/cpp_routines/kick_and_drift.cpp'],
                                 extra_compile_args=['-fopenmp',
                                                     '-ffast-math'],
                                 extra_link_args=['-lgomp'])

setup(ext_modules=[cpp_routines])
