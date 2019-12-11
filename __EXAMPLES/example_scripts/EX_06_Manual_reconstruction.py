import numpy as np

import tomo.machine as mch
import tomo.particles as parts
import tomo.cpp_routines.tomolib_wrappers as tlw
import tomo.tracking.tracking as tracking
import tomo.utils.tomo_input as tomoin
import tomo.utils.tomo_output as tomoout


def discrepancy(nbins, nprofs, dwaterfall):
    return np.sqrt(np.sum(dwaterfall**2)/(nbins * nprofs))

# Values retrieved from INDIVShavingC325.dat
dtbin = 9.999999999999999E-10

# This class is here to help shaping your waterfall from a fortran style
#  collection and format of measured raw data.
# This can be omitted by shaping the waterfall yourself.
frame_input_args = {
    'raw_data_path':       '../input_files/INDIVShavingC325.dat',
    'framecount':          150,
    'skip_frames':         0,
    'framelength':         1000,
    'dtbin':               dtbin,
    'skip_bins_start':     170,
    'skip_bins_end':       70,
    'rebin':               3 
}

frames = tomoin.Frames(**frame_input_args)
nprofiles = frames.nprofs()
nbins = frames.nbins()

# Machine and reconstruction parameters
machine_input_args = {
    'output_dir':          '/tmp/',
    'dtbin':               dtbin,
    'dturns':              5,
    'xat0':                334.00000000000006,
    'demax':               -1.E6,
    'filmstart':           0,
    'filmstop':            1,
    'filmstep':            1,
    'niter':               20,
    'snpt':                4,
    'full_pp_flag':        False,
    'beam_ref_frame':      0,
    'machine_ref_frame':   0,
    'vrf1':                2637.197030932989,
    'vrf1dot':             0.0,
    'vrf2':                0.0,
    'vrf2dot':             0.0,
    'h_num':               1,
    'h_ratio':             2.0,
    'phi12':               0.4007821253666541,
    'b0':                  0.15722,
    'bdot':                0.7949999999999925,
    'mean_orbit_radius':   25.0,
    'bending_radius':      8.239,
    'transitional_gamma':  4.1,
    'rest_energy':         0.93827231E9,
    'charge':              1,
    'self_field_flag':     False,
    'g_coupling':          0.0,
    'zwall_over_n':        0.0,
    'pickup_sensitivity':  0.36,
    'nprofiles':           nprofiles,
    'nbins':               nbins,
    'min_dt':              0.0,
    'max_dt':              dtbin * nbins}

machine = mch.Machine(**machine_input_args)

raw_data = np.genfromtxt(frames.raw_data_path, skip_header=98,
                         dtype=np.float32)
machine.values_at_turns()
waterfall = frames.to_waterfall(raw_data)

if machine.xat0 < 0 or machine.self_field_flag:
    msg = 'This example does not include fitting or self field tracking.'
    raise NotImplementedError(msg)

profiles = tomoin.raw_data_to_profiles(
                waterfall, machine, frames.rebin, frames.sampling_time)

tracker = tracking.Tracking(machine)
reconstruct_idx = machine.filmstart

# profile charge needed for fortran style output during tracking.
xp, yp = tracker.track(reconstruct_idx)

# Converts from physical units to phase space coordinates as bin numbers.
xp, yp = parts.physical_to_coords(
                xp, yp, machine, tracker.particles.xorigin,
                tracker.particles.dEbin)

# Filters out lost particles, transposes particle matrix, casts to np.int32.
xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)

# Waterfall must be normalized, and negatives
# reduced to zeroes befor reconstruction.
flat_profs = profiles.waterfall.copy() 
flat_profs = flat_profs.clip(0.0)
if not flat_profs.any():
    raise expt.WaterfallReducedToZero()
flat_profs /= np.sum(flat_profs, axis=1)[:, None]
flat_profs = np.ascontiguousarray(flat_profs.flatten()).astype(np.float64)

# Normalizing waterfall in profiles used for comparing
profiles.waterfall /= np.sum(profiles.waterfall, axis=1)[:, None]

# In order to use the cpp reconstruction routines, the profile
# arrays must be flattened. x-coordinates must be adjusted for this.
flat_points = xp.copy()
for i in range(machine.nprofiles):
    flat_points[:, i] += machine.nbins * i

# Reconstructing phase space
nparts = xp.shape[0]
weight = np.zeros(nparts)
rec_wf = np.zeros(profiles.waterfall.shape)

weight = tlw.back_project(weight, flat_points, flat_profs,
                          nparts, machine.nprofiles)

diff = []
for i in range(machine.niter):

    rec_wf = tlw.project(rec_wf, flat_points, weight, nparts,
                         machine.nprofiles, machine.nbins)
    
    rec_wf.clip(0.0)
    rec_wf /= np.sum(rec_wf, axis=1)[:, None]

    dwaterfall = profiles.waterfall - rec_wf
    rec_wf[:] = 0.0

    diff.append(discrepancy(machine.nbins, machine.nprofiles, dwaterfall))

    weight = tlw.back_project(weight, flat_points, dwaterfall.flatten(),
                              nparts, machine.nprofiles)

    print(f'Iteration: {i+1:3d}, discrepancy: {diff[-1]:3E}')


# Creating image for fortran style presentation of phase space. 
image = tomoout.create_phase_space_image(
            xp, yp, weight, machine.nbins, reconstruct_idx)
tomoout.show(image, diff, profiles.waterfall[reconstruct_idx])
