import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
lib = ctypes.cdll.LoadLibrary('cpp_files/map_weights.so')
lib.weight_factor_array.argtypes = [ndpointer(ctypes.c_double),
                                    ndpointer(ctypes.c_int),
                                    ndpointer(ctypes.c_int),
                                    ndpointer(ctypes.c_int),
                                    ndpointer(ctypes.c_int),
                                    ndpointer(ctypes.c_int),
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int]

xp = np.genfromtxt("/home/cgrindhe/cpp_test/xp.dat", dtype=np.double)
jmin = np.genfromtxt("/home/cgrindhe/cpp_test/jmin.dat", dtype=np.int32)
jmax = np.genfromtxt("/home/cgrindhe/cpp_test/jmax.dat", dtype=np.int32)
imin = 2
imax = 203
mapsi = np.zeros(406272, dtype=np.int32)
mapsi -= 1
mapsw = np.zeros(406272, dtype=np.int32)
maps = np.zeros(42025, dtype=np.int32)
array_length = 16
profile_length = 205
fmlistlength = 16
actmaps = 0
npt = 16
nr_of_arrays = 25392

isOut = lib.weight_factor_array(xp, jmin, jmax,
                                maps, mapsi, mapsw,
                                array_length, imin, imax,
                                npt, profile_length,
                                fmlistlength, nr_of_arrays,
                                actmaps)
a = np.load("/home/cgrindhe/tomo_v3/unit_tests/resources/C500MidPhaseNoise/mapsw.npy")
diff = a[nr_of_arrays: 2*nr_of_arrays].flatten() - mapsw
del a
print(str(diff.any()))
a = np.load("/home/cgrindhe/tomo_v3/unit_tests/resources/C500MidPhaseNoise/mapsi.npy")
diff = a[nr_of_arrays: 2*nr_of_arrays].flatten() - mapsi
print(str(diff.any()))
del a

print(mapsi[0:16])
print(mapsw[0:16])

# class CReconstruct:
#
#     def __init__(self, timespace, mapinfo):
#         self.timespace = timespace
#         self.mapinfo = mapinfo
#         self.maps = []
#         self.mapsi = []
#         self.mapweights = []
#         self.reversedweights = []
#         self.fmlistlength = 0
#
#     def reconstruct(self, timespace, mapinfo):
#         pass