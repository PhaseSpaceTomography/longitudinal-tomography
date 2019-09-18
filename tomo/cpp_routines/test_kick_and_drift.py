import ctypes as ct
import numpy as np
import os
import time as tm
import logging as log
import numpy.testing as nptest

def _get_pointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def _get_2d_pointer(arr2d):
    return (arr2d.__array_interface__['data'][0]
            + np.arange(arr2d.shape[0]) * arr2d.strides[0]).astype(np.uintp)

def main():

    os.system('clear')
    
    # Control panel
    # -----------------
    do_compile = True
    gpu = False
    niter = 1
    run_tests = True
    show_all_times = True
    #------------------
    
    cpu_com = 'g++ -fopenmp -march=native -ffast-math'
    gpu_com = 'pgc++ -acc -Minfo=accel -ta=nvidia'
    common_com = '-std=c++11 -shared -O3 -fPIC -o test.so kick_and_drift.cpp'

    # minfo can be accell and all
    
    if do_compile:
        print('Compiling!')
        os.system('rm test.so')
        if gpu:
            print('Compiling openACC')
            os.system(gpu_com + ' ' + common_com)
        else:
            print('Compiling openMP')
            os.system(cpu_com + ' ' + common_com)
    else:
        print('Using last compilation...')
    
    _tomolib_pth = os.path.dirname(os.path.realpath(__file__)) + '/test.so'
    
    if os.path.exists(_tomolib_pth):
        log.info(f'Loading test library: {_tomolib_pth}')
        lib = ct.CDLL(_tomolib_pth)
    else:
        error_msg = f'\n\nCould not find library at:\n{_tomolib_pth}\n'
        raise Exeption(error_msg)
    
    _double_ptr = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')

    k_and_d = lib.kick_and_drift
    k_and_d.argtypes = [_double_ptr,
                        _double_ptr,
                        np.ctypeslib.ndpointer(ct.c_double),
                        np.ctypeslib.ndpointer(ct.c_double),
                        np.ctypeslib.ndpointer(ct.c_double),
                        np.ctypeslib.ndpointer(ct.c_double),
                        np.ctypeslib.ndpointer(ct.c_double),
                        np.ctypeslib.ndpointer(ct.c_double),
                        np.ctypeslib.ndpointer(ct.c_double),
                        np.ctypeslib.ndpointer(ct.c_double),
                        ct.c_double,
                        ct.c_double,
                        ct.c_double,
                        ct.c_double,
                        ct.c_double,
                        ct.c_double,
                        ct.c_double,
                        ct.c_int,
                        ct.c_int,
                        ct.c_int]
    
    from tomo_v3.unit_tests.C500values import C500
    c500 = C500()
    cv = c500.values
    ca = c500.arrays
    
    # Getting actual values from c500
    # -----------------------------------------------
    x_origin = cv['xorigin']
    h_num = cv['h_num']
    omega_rev0 = ca['omegarev0']
    dtbin = cv['dtbin']
    phi0 = ca['phi0']
    yat0 = cv['yat0']
    deltaE0 = ca['deltaE0']
    dphase = ca['dphase']
    dturns = cv['dturns']
    phi12 = cv['phi12']
    hratio = cv['hratio']
    hnum = cv['h_num']
    dEbin = cv['debin']
    
    # Setting parameters for tracking
    # ------------------------------------------------
    # Start parameters:
    denergy = np.load('/afs/cern.ch/work/c/cgrindhe/tomography/out/denergy.npy')
    dphi = np.load('/afs/cern.ch/work/c/cgrindhe/tomography/out/dphi.npy')

    # other parameters
    nturns = cv['dturns'] * (cv['profile_count'] - 1)
    rfv1 = (cv['vrf1'] + cv['vrf1dot'] * ca['time_at_turn']) * cv['q']
    rfv2 = np.zeros(nturns + 1)
    
    nparts = len(denergy)
    
    xp = np.zeros((cv['profile_count'], nparts))
    yp = np.zeros((cv['profile_count'], nparts))
    xpp = _get_2d_pointer(xp)
    ypp = _get_2d_pointer(yp)
    
    print('Running c++ function...')
    
    # =======================================================================
    # RUNNING TEST
    # =======================================================================
    
    # -----------------------------------------------------------------------
    # runnin script....
    # -----------------------------------------------------------------------   
    
    run_times = np.zeros(niter, dtype=float)
   
    for i in range(niter):
        t0 = tm.perf_counter()
        k_and_d(xpp, ypp, denergy, dphi, rfv1, rfv2, phi0, deltaE0,
                omega_rev0, dphase, phi12, hratio, hnum, dtbin, x_origin,
                dEbin, yat0, dturns, nturns, nparts)
        t1 = tm.perf_counter()
        run_times[i] = t1 - t0
    
    total_time = np.sum(run_times)

    print('\nExecution info:\n---------------------')
    print(f'Compile: {do_compile}')
    print(f'GPU: {gpu}')
    print(f'Number of runs: {niter}')
    print(f'Run tests: {run_tests}')

    if run_tests:
        print('\nTesting output:\n---------------------')
        output_dir = '/afs/cern.ch/work/c/cgrindhe/tomography/out'
    
        print('testing xp...')
        old_xp = np.load(output_dir + '/c_xp.npy')
        nptest.assert_almost_equal(xp, old_xp,
                                   err_msg='xp was not '\
                                           'calculated correctly.')
        print('OK!')
    
        print('testing yp...')
        old_yp = np.load(output_dir + '/c_yp.npy')
        nptest.assert_almost_equal(yp, old_yp,
                                   err_msg='yp was not '\
                                           'calculated correctly.')
        print('OK!')
    
    print('\nTiming:\n---------------------')
    print(f'time spent: {total_time}')
    print(f'average time: {total_time / niter}')

    if show_all_times:
        for i, t in enumerate(run_times):
            print(f'run #{i+1}: {t}')

    # =======================================================================

main()
print()