import ctypes as ct
import numpy as np
import os
from utils.exceptions import LibraryNotFound
import logging as log

log.basicConfig(level=log.DEBUG)

_tomolib_pth = os.path.dirname(os.path.realpath(__file__)) + '/tomolib.so'

if os.path.exists(_tomolib_pth):
    log.info(f'Loading C++ library: {_tomolib_pth}')
    _tomolib = ct.CDLL(_tomolib_pth)
else:
    raise LibraryNotFound(f'Could not find library at {_tomolib_pth}')

_double_ptr = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')

_back_project = _tomolib.back_project
_back_project.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                          _double_ptr, np.ctypeslib.ndpointer(ct.c_double)]
_back_project.restypes = None

_proj = _tomolib.project
_proj.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                  _double_ptr, np.ctypeslib.ndpointer(ct.c_double)]
_proj.restypes = None

# =============================================================
# Functions for paricle tracking
# =============================================================


def kick(parameters, denergy, dphi, rfv1, rfv2, nr_part, turn):
    _tomolib.new_kick(_get_pointer(dphi),
                      _get_pointer(denergy),
                      ct.c_double(rfv1[turn]),
                      ct.c_double(rfv2[turn]),
                      ct.c_double(parameters.phi0[turn]),
                      ct.c_double(parameters.phi12),
                      ct.c_double(parameters.h_ratio),
                      ct.c_int(nr_part),
                      ct.c_double(parameters.deltaE0[turn]))
    return denergy


def drift(denergy, dphi, dphase, nr_part, turn):
    _tomolib.new_drift(_get_pointer(dphi),
                       _get_pointer(denergy),
                       ct.c_double(dphase[turn]),
                       ct.c_int(nr_part))
    return dphi

# =============================================================
# Functions for phase space reconstruction
# =============================================================


def back_project(weights, flat_points, flat_profiles, nparts, nprofs):
    _back_project(weights, _get_2d_pointer(flat_points),
                  flat_profiles, nparts, nprofs)
    return weights


def project(rec_ps, flat_points, weights, nparts, nprofs, nbins):
    rec_ps = np.ascontiguousarray(rec_ps.flatten())
    _proj(rec_ps, _get_2d_pointer(flat_points), weights, nparts, nprofs)
    rec_ps = rec_ps.reshape((nprofs, nbins))
    return rec_ps

# =============================================================
# Utilities
# =============================================================


def _get_pointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def _get_2d_pointer(arr2d):
    return (arr2d.__array_interface__['data'][0]
            + np.arange(arr2d.shape[0]) * arr2d.strides[0]).astype(np.uintp)
