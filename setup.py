from setuptools import setup, Extension

cpp_routines = Extension('tomo.cpp_routines.tomolib',
                         sources=['tomo/cpp_routines/reconstruct.cpp',
                                  'tomo/cpp_routines/kick_and_drift.cpp'],
                         extra_compile_args=['-fopenmp',
                                             '-fPIC',
                                             '-std=c++11',
                                             '-march=native',
                                             '-ffast-math'],
                         extra_link_args=['-lgomp'])

setup(ext_modules=[cpp_routines])
