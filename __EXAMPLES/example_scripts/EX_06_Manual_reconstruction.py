import os

import numpy as np

import tomo.cpp_routines.tomolib_wrappers as tlw
import tomo.utils.tomo_output as tomoout


def discrepancy(nbins, nprofs, dwaterfall):
    return np.sqrt(np.sum(dwaterfall ** 2) / (nbins * nprofs))


raise NotImplementedError('The resources required for this example has '
                          'not yet been implemented properly.')
# this_dir = os.path.realpath(os.path.dirname(__file__))
# resource_dir = os.path.join(this_dir, 'resources')
# 
# xp = np.load(os.path.join(resource_dir, 'INDIVShavingC325_xcoords.npy'))
# yp = np.load(os.path.join(resource_dir, 'INDIVShavingC325_ycoords.npy'))
# waterfall = np.load(os.path.join(
#     resource_dir, 'INDIVShavingC325_waterfall.npy'))

niterations = 20
nprofs = waterfall.shape[0]
nbins = waterfall.shape[1]
nparts = xp.shape[0]
rec_tframe = 0

# Remove comment to track using tomo routine:
# ------------------------------------------------------
# import tomo.tomography.tomography_cpp as tomography
# import sys
# tomo = tomography.TomographyCpp(waterfall, xp)
# weight = tomo.run(niter=niterations)
# image = tomoout.create_phase_space_image(
#             xp, yp, weight, nbins, rec_tframe)
# tomoout.show(image, tomo.diff, waterfall[rec_tframe])
# sys.exit()
# ------------------------------------------------------

# Waterfall must be normalized, and negatives
# reduced to zeroes befor reconstruction.
flat_profs = waterfall.copy()
flat_profs = flat_profs.clip(0.0)

flat_profs /= np.sum(flat_profs, axis=1)[:, None]
flat_profs = np.ascontiguousarray(flat_profs.flatten()).astype(np.float64)

# Normalizing waterfall in profiles used for comparing
waterfall /= np.sum(waterfall, axis=1)[:, None]

# In order to use the cpp reconstruction routines, the profile
# arrays must be flattened. x-coordinates must be adjusted for this.
flat_points = xp.copy()
for i in range(nprofs):
    flat_points[:, i] += nbins * i

# Reconstructing phase space
weight = np.zeros(nparts)
rec_wf = np.zeros(waterfall.shape)

# Initial estimation of weight factors using (flattended) measured profiles. 
weight = tlw.back_project(weight, flat_points, flat_profs,
                          nparts, nprofs)
weight = weight.clip(0.0)

diff = []
for i in range(niterations):
    # Projection from phase space to time projections
    rec_wf = tlw.project(rec_wf, flat_points, weight, nparts,
                         nprofs, nbins)

    # Normalizing reconstructed waterfall
    rec_wf /= np.sum(rec_wf, axis=1)[:, None]

    # Finding difference between measured and reconstructed waterfall
    dwaterfall = waterfall - rec_wf

    # Setting to zero for next round
    rec_wf[:] = 0.0

    # Calculating discrepancy
    diff.append(np.sqrt(np.sum(dwaterfall ** 2) / (nbins * nprofs)))

    # Back projecting using the difference between measured and rec. waterfall 
    weight = tlw.back_project(weight, flat_points, dwaterfall.flatten(),
                              nparts, nprofs)
    weight = weight.clip(0.0)

    print(f'Iteration: {i:3d}, discrepancy: {diff[-1]:3E}')

# Finding last discrepancy...
rec_wf = tlw.project(rec_wf, flat_points, weight, nparts, nprofs, nbins)
rec_wf /= np.sum(rec_wf, axis=1)[:, None]
dwaterfall = waterfall - rec_wf
diff.append(np.sqrt(np.sum(dwaterfall ** 2) / (nbins * nprofs)))
print(f'Iteration: {i + 1:3d}, discrepancy: {diff[-1]:3E}')

# Creating image for fortran style presentation of phase space
image = tomoout.create_phase_space_image(
    xp, yp, weight, nbins, rec_tframe)
tomoout.show(image, diff, waterfall[rec_tframe])
