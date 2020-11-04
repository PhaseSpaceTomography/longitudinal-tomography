import os

import numpy as np
import tomo.cpp_routines.tomolib_wrappers as tlw
import tomo.tracking.machine as mch
import tomo.tracking.particles as parts
import tomo.tracking.tracking as tracking
import tomo.utils.tomo_input as tomoin
import tomo.utils.tomo_output as tomoout


def discrepancy(nbins, nprofs, dwaterfall):
    return np.sqrt(np.sum(dwaterfall ** 2) / (nbins * nprofs))


# -----------------------------------------------------------------------------
# Data loading or generation, not part of the example. Skip ahead.
# -----------------------------------------------------------------------------
this_dir = os.path.realpath(os.path.dirname(__file__))
resource_dir = os.path.join(this_dir, 'resources')

xp_file = os.path.join(resource_dir, 'INDIVShavingC325_xcoords.npy')
yp_file = os.path.join(resource_dir, 'INDIVShavingC325_ycoords.npy')
waterfall_file = os.path.join(resource_dir, 'INDIVShavingC325_waterfall.npy')

if os.path.exists(xp_file) and os.path.isfile(xp_file):
    xp = np.load(xp_file)
    yp = np.load(yp_file)
    waterfall = np.load(waterfall_file)
else:
    # construct pre-tracked data, this routine is not part of the example
    # the saved numpy arrays take up too much space in the repo, instead
    # they are generated when needed
    ex_dir = os.path.split(this_dir)[0]
    in_file_pth = os.path.join(ex_dir, 'input_files', 'INDIVShavingC325.dat')

    with open(in_file_pth, 'r') as file:
        raw_params, raw_data = tomoin._split_input(file.readlines())

    machine, frames = tomoin.txt_input_to_machine(raw_params)
    machine.values_at_turns()
    measured_waterfall = frames.to_waterfall(raw_data)

    profiles = tomoin.raw_data_to_profiles(
        measured_waterfall, machine, frames.rebin, frames.sampling_time)
    profiles.calc_profilecharge()

    if profiles.machine.synch_part_x < 0:
        fit_info = dtreat.fit_synch_part_x(profiles)
        machine.load_fitted_synch_part_x_ftn(fit_info)

    reconstr_idx = machine.filmstart

    # Tracking...
    tracker = tracking.Tracking(machine)

    xp, yp = tracker.track(reconstr_idx)

    # Converting from physical coordinates ([rad], [eV])
    # to phase space coordinates.
    if not tracker.self_field_flag:
        xp, yp = parts.physical_to_coords(
            xp, yp, machine, tracker.particles.xorigin,
            tracker.particles.dEbin)

    # Filters out lost particles, transposes particle matrix,
    # casts to np.int32.
    xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)

    waterfall = profiles.waterfall

    os.makedirs(resource_dir, exist_ok=True)
    np.save(xp_file, xp)
    np.save(yp_file, yp)
    np.save(waterfall_file, waterfall)

# -----------------------------------------------------------------------------
# Begin main example
# -----------------------------------------------------------------------------

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
# reduced to zeroes before reconstruction.
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

# Initial estimation of weight factors using (flattened) measured profiles.
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
