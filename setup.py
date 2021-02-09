from setuptools import setup, Extension
from os import path

HERE = path.split(path.abspath(__file__))[0]

cpp_routines = Extension('longitudinal_tomography.cpp_routines.tomolib',
                         sources=['longitudinal_tomography/cpp_routines/reconstruct.cpp',
                                  'longitudinal_tomography/cpp_routines/kick_and_drift.cpp'],
                         extra_compile_args=['-fopenmp',
                                             '-fPIC',
                                             '-std=c++11',
                                             '-ffast-math'],
                         extra_link_args=['-lgomp'])

setup(ext_modules=[cpp_routines])
