import ctypes as ct
import numpy as np
import os
import sys
import time as tm
import logging as log
import numpy.testing as nptest
import matplotlib.pyplot as plt

def _get_pointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def _get_2d_pointer(arr2d):
    return (arr2d.__array_interface__['data'][0]
            + np.arange(arr2d.shape[0]) * arr2d.strides[0]).astype(np.uintp)


def compile(gpu):
    name_so = 'test_rec.so'
    compile_file = 'reconstruct.cpp'

    cpu_com = 'g++ -fopenmp -march=native -ffast-math'
    gpu_com = 'pgc++ -acc -Minfo=accel -ta=nvidia'
    common_com = f'-std=c++11 -shared -O3 -fPIC -o {name_so} {compile_file}'
    command = ''

    os.system(f'rm {name_so}')
    if gpu:
        print('Compiling openACC')
        command = gpu_com + ' ' + common_com
    else:
        print('Compiling openMP')
        command = cpu_com + ' ' + common_com
    os.system(command)

def get_lib():
    _tomolib_pth = os.path.dirname(os.path.realpath(__file__)) + '/test_rec.so'
    
    if os.path.exists(_tomolib_pth):
        log.info(f'Loading test library: {_tomolib_pth}')
        lib = ct.CDLL(_tomolib_pth)
    else:
        error_msg = f'\n\nCould not find library at:\n{_tomolib_pth}\n'
        raise Exeption(error_msg)
    return lib

def set_up_function(lib):
    double_ptr = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')
    reconstruct = lib.reconstruct
    reconstruct.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                            double_ptr,
                            np.ctypeslib.ndpointer(ct.c_double),
                            ct.c_int,
                            ct.c_int,
                            ct.c_int,
                            ct.c_int]
    return reconstruct

def check_output(weights, xp, yp, profiles, nbins, film=0):
    phase_space = np.zeros((nbins, nbins))

    for x, y, w in zip(xp[:, film], yp[:, film], weights):
        phase_space[x, y] += w

    # Surpressing negative numbers
    phase_space = phase_space.clip(0.0)

    # Normalizing
    phase_space /= np.sum(phase_space)
    
    plt.subplot(211)
    plt.imshow(phase_space.T, cmap='hot', interpolation='nearest', origin='lower')
    plt.subplot(212)
    plt.plot(profiles[film])
    plt.plot(np.sum(phase_space.T, axis=0))
    plt.show()

def main():
    os.system('clear')

    # Flags and constrols
    # ------------------
    niter = 20
    do_compile = True
    use_gpu_flg = False
    show_image = True

    if do_compile:
        print('Compiling!')
        compile(gpu=use_gpu_flg)

    lib = get_lib()
    reconstruct = set_up_function(lib)

    # test_func = lib.test_func
    # test_func.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
    #                       ct.c_int,
    #                       ct.c_int]
    # lib.test_func.restype = np.ctypeslib.ndpointer(dtype=ct.c_double, shape=(100,))

    # profiles = ca['profiles']
    # profiles = np.ascontiguousarray(profiles.flatten().astype(ct.c_double))

    # print(profiles[:10])

    # test = test_func(profiles, 0, 0)

    # sys.exit('\nEND TEST')


    from tomo_v3.unit_tests.C500values import C500
    c500 = C500()
    cv = c500.values
    ca = c500.arrays

    profiles = ca['profiles']
    profiles = np.ascontiguousarray(profiles.flatten().astype(ct.c_double))

    xp = np.load('/afs/cern.ch/work/c/cgrindhe/'\
                 'tomography/out/py_xp.npy')
    xp = np.ascontiguousarray(xp.T.astype(ct.c_int))

    yp = np.load('/afs/cern.ch/work/c/cgrindhe/'\
                 'tomography/out/py_yp.npy')
    yp = np.ascontiguousarray(yp.T.astype(ct.c_int))

    nbins = cv['reb_profile_length']
    nparts = xp.shape[0]
    nprofs = cv['profile_count']

    weights = np.ascontiguousarray(np.zeros(nparts, dtype=ct.c_double))

    print('\n===============\nRunning C++!\n===============\n')
    t0 = tm.perf_counter()
    reconstruct(weights, _get_2d_pointer(xp), profiles, niter, nbins, nparts, nprofs)
    t1 = tm.perf_counter()
    print('\n===============\nFinito finale!\n===============')

    print('\n-------------------\nTime spent\n-------------------\n')
    print(f'Time spent: {t1 - t0} s')

    # print('loading original values...')
    # original_w = np.load('/afs/cern.ch/work/c/cgrindhe/tomography/out/weight0.npy')
    # original_rec = np.load('/afs/cern.ch/work/c/cgrindhe/tomography/out/rec0.npy')

    # diff = ca['profiles'] - original_rec

    if show_image:
        check_output(weights, xp, yp, ca['profiles'], nbins)

main()
print()
