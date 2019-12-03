import numpy as np
import matplotlib.pyplot as plt
import sys

import tomo.machine as mch
import tomo.profiles as prf
import tomo.tracking.tracking as trc
import tomo.tomography.tomography_cpp as tmo
import tomo.fit as fit

margs = {
         'output_dir':          None,
         'dtbin':               float(250E-9 / 500),
         'dturns':              2,
         'xat0':                251.0,
         'demax':               -1.E6,
         'filmstart':           0,
         'filmstop':            1,
         'filmstep':            1,
         'niter':               20,
         'snpt':                4,
         'full_pp_flag':        False,
         'beam_ref_frame':      0,
         'machine_ref_frame':   0,
         'vrf1':                0.0,
         'vrf1dot':             0.0,
         'vrf2':                0.0,
         'vrf2dot':             0.0,
         'h_num':               1,
         'h_ratio':             2,
         'phi12':               0.3116495273194016,
         'b0':                  0.236,
         'bdot':                0.0,
         'mean_orbit_radius':   25.0,
         'bending_radius':      8.239,
         'transitional_gamma':  1.0 / np.sqrt(0.06016884),
         'rest_energy':         0.93827231E9,
         'charge':              1,
         'self_field_flag':     False,
         'g_coupling':          0.0,
         'zwall_over_n':        0.0,
         'pickup_sensitivity':  0.36,
         'nprofiles':           70,
         'nbins':               500,
         'min_dt':              None,
         'max_dt':              None
        }

sim_dat_dir = '/afs/cern.ch/work/c/cgrindhe/tomography/'\
              'lab/debunching/simulated_data'

machine = mch.Machine(**margs)
machine.values_at_turns()

waterfall = np.load(f'{sim_dat_dir}/profiles.npy')
profiles = prf.Profiles(machine, machine.dtbin, waterfall)



dts = np.load(f'{sim_dat_dir}/dt.npy')
dEs = np.load(f'{sim_dat_dir}/dE.npy')

icoords = (dts[0], dEs[0])

tracker = trc.Tracking(machine)
xp, yp = tracker.track(initial_coordinates=icoords)


