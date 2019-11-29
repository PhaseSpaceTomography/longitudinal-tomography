import ctypes as ct
import numpy as np
import os
import logging as log

from ..utils import exceptions as expt

log.basicConfig(level=log.INFO)

_tomolib_pth = os.path.dirname(os.path.realpath(__file__)) + '/tomolib.so'

if os.path.exists(_tomolib_pth):
    log.info(f'Loading C++ library: {_tomolib_pth}')
    _tomolib = ct.CDLL(_tomolib_pth)
else:
    error_msg = f'\n\nCould not find library at:\n{_tomolib_pth}\n' \
                f'\n- Try to run compile.py in the tomo directory\n'
    raise expt.LibraryNotFound(error_msg)

_double_ptr = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')

# Kick and drift (cpu version)
_k_and_d = _tomolib.kick_and_drift
_k_and_d.argtypes = [_double_ptr,
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

# Kick and drift (gpu version)
_k_and_d_gpu = _tomolib.kick_and_drift_gpu
_k_and_d_gpu.argtypes = _k_and_d.argtypes

# Reconstruction routine (flat version)
_reconstruct = _tomolib.reconstruct
_reconstruct.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                         _double_ptr,
                         np.ctypeslib.ndpointer(ct.c_double),
                         np.ctypeslib.ndpointer(ct.c_double),
                         ct.c_int, ct.c_int,
                         ct.c_int, ct.c_int]

# Back_project (flat version)
_back_project = _tomolib.back_project
_back_project.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                          _double_ptr, np.ctypeslib.ndpointer(ct.c_double)]
_back_project.restypes = None

# Project (flat version)
_proj = _tomolib.project
_proj.argtypes = [np.ctypeslib.ndpointer(ct.c_double),
                  _double_ptr, np.ctypeslib.ndpointer(ct.c_double)]
_proj.restypes = None

# =============================================================
# Functions for paricle tracking
# =============================================================


def kick(parameters, denergy, dphi, rfv1, rfv2, nr_part, turn, up=True):
    args = (_get_pointer(dphi),
            _get_pointer(denergy),
            ct.c_double(rfv1[turn]),
            ct.c_double(rfv2[turn]),
            ct.c_double(parameters.phi0[turn]),
            ct.c_double(parameters.phi12),
            ct.c_double(parameters.h_ratio),
            ct.c_int(nr_part),
            ct.c_double(parameters.deltaE0[turn]))
    if up:
        _tomolib.kick_up(*args)
    else:
        _tomolib.kick_down(*args)
    return denergy


def drift(denergy, dphi, dphase, nr_part, turn, up=True):
    args = (_get_pointer(dphi),
            _get_pointer(denergy),
            ct.c_double(dphase[turn]),
            ct.c_int(nr_part))
    if up:
        _tomolib.drift_up(*args)
    else:
        _tomolib.drift_down(*args)
    return dphi

def kick_and_drift(xp, yp, denergy, dphi, rfv1, rfv2, phi0,
                   deltaE0, omega_rev0, dphase, phi12, hratio,
                   hnum, dtbin, xorigin, dEbin, yat0, dturns,
                   nturns, npts, gpu_flag=False):
    args = (_get_2d_pointer(xp), _get_2d_pointer(yp), denergy, dphi,
             rfv1, rfv2, phi0, deltaE0, omega_rev0, dphase, phi12, hratio,
             hnum, dtbin, xorigin, dEbin, yat0, dturns, nturns, npts)
    
    if gpu_flag:
        # Info here might not be true.
        # Find a way to choose from either CPU or GPU version.
        #  - Makefile?
        log.info('Tracking particles using GPU.')
        _k_and_d_gpu(*args)
    else:
        log.info('Tracking particles using CPU.')
        _k_and_d(*args)

    return xp, yp


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


def reconstruct(weights, xp, flat_profiles, discr,
                niter, nbins, npart, nprof):
    _reconstruct(weights, _get_2d_pointer(xp), flat_profiles,
                 discr, niter, nbins, npart, nprof)
    return weights


# =============================================================
# Utilities
# =============================================================

def _get_pointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def _get_2d_pointer(arr2d):
    return (arr2d.__array_interface__['data'][0]
            + np.arange(arr2d.shape[0]) * arr2d.strides[0]).astype(np.uintp)
