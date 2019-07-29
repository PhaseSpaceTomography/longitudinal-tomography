import ctypes as ct
import os

tomolib_pth = os.path.dirname(os.path.realpath(__file__)) + '/tomolib.so'
tomolib = ct.CDLL(tomolib_pth)


def kick(parameters, denergy, dphi, rfv1, rfv2, nr_part, turn):
    tomolib.new_kick(_get_pointer(dphi),
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
    tomolib.new_drift(_get_pointer(dphi),
                      _get_pointer(denergy),
                      ct.c_double(dphase[turn]),
                      ct.c_int(nr_part))
    return dphi


def _get_pointer(x):
    return x.ctypes.data_as(ct.c_void_p)
