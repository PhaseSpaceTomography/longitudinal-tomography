import ctypes as ct
import numpy as np
import os
import sys
import time as tm
import logging as log
import numpy.testing as nptest
import matplotlib.pyplot as plt

def main():
    os.system('clear')

    # Flags and constrols
    # ------------------
    nruns = 1
    do_compile = True
    use_gpu = True
    show_all_times = True
    show_image = False
    test = True

    if do_compile:
        print('Compiling!')
        compile(gpu=use_gpu)
    else:
        print('Running on last compilation')

    lib = get_lib()
    
    reconstruct = set_up_function(lib)
    # reconstruct = set_up_function_no_flat(lib)

    from tomo_v3.unit_tests.C500values import C500
    c500 = C500()
    cv = c500.values
    ca = c500.arrays

    niter = 20

    profiles = ca['profiles']
    # profiles = np.ascontiguousarray(profiles.astype(ct.c_double))
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
    dt = np.zeros(nruns)
    for i in range(nruns):
        print(f'Run: {i}')
        weights[:] = 0
        t0 = tm.perf_counter()
        reconstruct(weights, _get_2d_pointer(xp), profiles, niter, nbins, nparts, nprofs)
        dt[i] = tm.perf_counter() - t0
    print('\n===============\nFinito finale!\n===============')

    print('\n-------------------\nTime spent\n-------------------\n')
    print(f'Time spent: {np.sum(dt)}s')
    print(f'Average time: {np.sum(dt) / nruns}s')
    print('Iteration times:')
    if show_all_times:
        for i, time in enumerate(dt):
            print(f'\t{i}: {time}')

    py_im = np.load('/afs/cern.ch/work/c/cgrindhe/tomography/out/py_image0.npy')

    if test:
        # test_first_weight(weights)
        test_output(weights, py_im, xp, yp, nbins)

    if show_image:
        show_output(weights, xp, yp, ca['profiles'], py_im, nbins)

def _get_pointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def _get_2d_pointer(arr2d):
    return (arr2d.__array_interface__['data'][0]
            + np.arange(arr2d.shape[0]) * arr2d.strides[0]).astype(np.uintp)


def compile(gpu):
    name_so = 'test_rec.so'
    compile_file = 'reconstruct_old_alg.cpp'

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
                            ct.c_int, ct.c_int,
                            ct.c_int, ct.c_int]
    return reconstruct

def set_up_function_no_flat(lib):
    double_ptr = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')
    reconstruct = lib.reconstruct
    reconstruct.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                            double_ptr,
                            double_ptr,
                            ct.c_int, ct.c_int,
                            ct.c_int, ct.c_int]
    return reconstruct

def show_output(weights, xp, yp, profiles, py_im, nbins, film=0):
    phase_space = np.zeros((nbins, nbins))

    for x, y, w in zip(xp[:, film], yp[:, film], weights):
        phase_space[x, y] += w

    # Surpressing negative numbers
    phase_space = phase_space.clip(0.0)

    # Normalizing
    phase_space /= np.sum(phase_space)
    
    plt.subplot(221)
    plt.imshow(phase_space.T, cmap='hot', interpolation='nearest', origin='lower')
    plt.subplot(222)
    plt.imshow(py_im.T, cmap='hot', interpolation='nearest', origin='lower')
    plt.subplot(223)
    plt.plot(profiles[film], label='measured')
    plt.plot(np.sum(phase_space.T, axis=0), label='c++')
    plt.plot(np.sum(py_im.T, axis=0), label='py')
    plt.gca().legend()
    plt.subplot(224)
    plt.imshow(np.abs(py_im.T - phase_space.T), cmap='hot',
               interpolation='nearest', origin='lower')
    plt.show()

def test_output(weights, py_im, xp, yp, nbins, film=0):
    phase_space = np.zeros((nbins, nbins))

    for x, y, w in zip(xp[:, film], yp[:, film], weights):
        phase_space[x, y] += w

    # Surpressing negative numbers
    phase_space = phase_space.clip(0.0)

    # Normalizing
    phase_space /= np.sum(phase_space)

    # Testing
    print("Testing output...")
    nptest.assert_almost_equal(phase_space, py_im)
    print("OK!")

def test_first_weight(weight):
    fw = np.load('/afs/cern.ch/work/c/cgrindhe/tomography/out/weight0.npy')
    print("\nTesting first weight...")
    nptest.assert_almost_equal(weight, fw)
    print("OK!\n")    

main()
print()
