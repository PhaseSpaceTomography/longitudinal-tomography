# General imports
import os

import matplotlib.pyplot as plt
import numpy as np

# Tomo imports
import longitudinal_tomography.utils.tomo_data as tdata
import longitudinal_tomography.utils.tomo_input as tomoin

ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', 'flatTopINDIVRotate2.dat')

# Raw data follows the parameter lines in the input file
raw_data = np.loadtxt(in_file_pth, skiprows=tomoin.PARAMETER_LENGTH)

# All values retrieved from flatTopINDIVRotate2.dat
frames = tdata.Frames(
    nframes=390,
    nbins_frame=400,
    skip_frames=0,
    skip_bins_start=0,
    skip_bins_end=0,
    rebin=3,
    sampling_time=4.999999999999999E-10,
    raw_data=raw_data)

# filmstart, filmstop, beam_ref_frame and machine_ref_frame are 0-based, one
# less than in the input file.
parameters = tdata.Parameters(
    output_dir='/tmp/',
    dturns=9,
    min_dt_bin=0,
    max_dt_bin=0,
    synch_part_x=202.39800000000002,
    demax=-1.E6,
    filmstart=0,
    filmstop=0,
    filmstep=1,
    niter=20,
    snpt=4,
    full_pp_flag=False,
    beam_ref_frame=0,
    machine_ref_frame=0,
    vrf1=7953.782859828863,
    vrf1dot=0.0,
    vrf2=0.0,
    vrf2dot=0.0,
    h_num=1,
    h_ratio=2.0,
    phi12=8.135620868822723E-4,
    b0=0.86129,
    bdot=0.004999999999991972,
    mean_orbit_rad=25.0,
    bending_rad=8.239,
    trans_gamma=4.1,
    rest_energy=0.93827231E9,
    charge=1,
    self_field_flag=False,
    g_coupling=0.0,
    zwall_over_n=0.0,
    pickup_sensitivity=0.36)

tomo_data = tdata.TomographyData(parameters=parameters, frames=frames)

tRange, ERange, density = tomo_data.run()

# %%

vmin = np.min(density[density > 0])
vmax = np.max(density)

plt.contourf(tRange * 1E9, ERange / 1E6, density.T,
             levels=np.linspace(vmin, vmax, 50), cmap='Oranges')
plt.xlabel('dt (ns)')
plt.ylabel('dE (MeV)')
plt.show()
