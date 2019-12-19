import numpy as np
import os

import tomo.particles as parts
import tomo.tracking.machine as mch
import tomo.tomography.tomography_cpp as tomography
import tomo.tracking.tracking as tracking
import tomo.utils.tomo_input as tomoin
import tomo.utils.tomo_output as tomoout

ex_dir = os.path.realpath(os.path.dirname(__file__)).split('/')[:-1]
in_file_pth = '/'.join(ex_dir + ['/input_files/INDIVShavingC325.dat'])

# All values retrieved from INDIVShavingC325.dat
dtbin = 9.999999999999999E-10

# This class is here to help shaping your waterfall from a fortran style
#  collection and format of measured raw data.
# This can be omitted by shaping the waterfall yourself.
frame_input_args = {
    'raw_data_path':       in_file_pth,
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
machine_args = {
    'output_dir':          '/tmp/',
    'dtbin':               dtbin,
    'dturns':              5,
    'synch_part_x':        334.00000000000006,
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
    'mean_orbit_rad':      25.0,
    'bending_rad':         8.239,
    'trans_gamma':         4.1,
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

machine = mch.Machine(**machine_args)

raw_data = np.genfromtxt(frames.raw_data_path, skip_header=98,
                         dtype=np.float32)
machine.values_at_turns()
measured_waterfall = frames.to_waterfall(raw_data)

if machine.synch_part_x < 0 or machine.self_field_flag:
    msg = 'This example does not include fitting or self field tracking.'
    raise NotImplementedError(msg)

profiles = tomoin.raw_data_to_profiles(
                measured_waterfall, machine,
                frames.rebin, frames.sampling_time)

tracker = tracking.Tracking(machine)
reconstruct_idx = machine.filmstart

# profile charge needed for fortran style output during tracking.
profiles.calc_profilecharge()
xp, yp = tracker.track(reconstruct_idx)

# Converts from physical units to phase space coordinates as bin numbers.
xp, yp = parts.physical_to_coords(
                xp, yp, machine, tracker.particles.xorigin,
                tracker.particles.dEbin)

# Filters out lost particles, transposes particle matrix, casts to np.int32.
xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)

# Reconstructing phase space
tomo = tomography.TomographyCpp(profiles.waterfall, xp)
weight = tomo.run(niter=machine.niter)

# Creating image for fortran style presentation of phase space. 
image = tomoout.create_phase_space_image(
            xp, yp, weight, machine.nbins, reconstruct_idx)
tomoout.show(image, tomo.diff, profiles.waterfall[reconstruct_idx])
