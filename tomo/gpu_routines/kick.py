import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath

import numpy as np

# The types of the used arrays should be set when used as input.
# I think they should all be set to np.float32, but it might work with
#  np.float64 as well. This must be tested.
def kick(dphi, denergy, rfv1, rfv2, phi0, phi12,
         hratio, nr_particles, acc_kick):
    dphi_gpu = gpuarray.to_gpu(dphi)
    denegy_gpu = gpuarray.to_gpu(denergy)
    rfv1_gpu = gpuarray.to_gpu(rfv1)
    rfv2_gpu = gpuarray.to_gpu(rfv2)

    denegy_gpu += rfv1_gpu * cumath.sin(dphi_gpu + phi0)
                  + rfv2_gpu * cumath.sin(hratio * (dphi_gpu + phi0 - phi12))

    # WRITE THE REST!

