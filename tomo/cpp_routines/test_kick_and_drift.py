import ctypes as ct
import numpy as np
import os
import time as tm
import logging as log
import numpy.testing as nptest

os.system('clear')

do_compile = True
if do_compile:
    print('Compiling!')
    os.system('rm test.so')
    os.system('g++ -fopenmp -std=c++11 -shared -O3 -march=native -fPIC -ffast-math -o test.so kick_and_drift.cpp')
else:
    print('Using last compilation...')

_tomolib_pth = os.path.dirname(os.path.realpath(__file__)) + '/test.so'

if os.path.exists(_tomolib_pth):
    log.info(f'Loading test library: {_tomolib_pth}')
    lib = ct.CDLL(_tomolib_pth)
else:
    error_msg = f'\n\nCould not find library at:\n{_tomolib_pth}\n'
    raise Exeption(error_msg)

_double_ptr = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C') # May be a problem

def _get_pointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def _get_2d_pointer(arr2d):
    return (arr2d.__array_interface__['data'][0]
            + np.arange(arr2d.shape[0]) * arr2d.strides[0]).astype(np.uintp)


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
niter = 1

t0 = tm.perf_counter()

for i in range(niter):
    k_and_d(xpp, ypp, denergy, dphi, rfv1, rfv2, phi0, deltaE0,
            omega_rev0, dphase, phi12, hratio, hnum, dtbin, x_origin,
            dEbin, yat0, dturns, nturns, nparts)

t1 = tm.perf_counter()
dt = t1 - t0

# -----------------------------------------------------------------------
# Asserting output...
# -----------------------------------------------------------------------
print('Output:\n--------------')
output_dir = '/afs/cern.ch/work/c/cgrindhe/tomography/out'

print('testing xp...')
old_xp = np.load(output_dir + '/c_xp.npy')
nptest.assert_almost_equal(xp, old_xp,
                           err_msg='xp was not '\
                                   'calculated correctly.')
print('OK!')

print('testing xp...')
old_yp = np.load(output_dir + '/c_yp.npy')
nptest.assert_almost_equal(yp, old_yp,
                           err_msg='yp was not '\
                                   'calculated correctly.')
print('OK!')

print('\nTiming:\n----------------')
print(f'time spent: {dt}')
print(f'average time: {dt / niter}')

# =======================================================================

