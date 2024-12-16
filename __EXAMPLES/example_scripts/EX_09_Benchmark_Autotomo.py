"""
Example implementation to create benchmarks for the GPU implementation.
First, bunch profiles will be created with BLonD using different
parameters. The focus will be on parameters for the PS Booster.
Second, the tracking and the reconstruction will be executed.
The runtime will be measured using Pyprof and saved.
"""

from pyprof import timing
import time

start_time = time.time()

timing.start_timing('import_packages')

# General imports
import numpy as np
import matplotlib.pyplot as plt
import os

# BLonD imports
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import matched_from_distribution_function
from blond.beam.profile import Profile, CutOptions
from blond.monitors.monitors import SlicesMonitor

# Tomography imports
from longitudinal_tomography.data import data_treatment as dtreat
import longitudinal_tomography.tracking.machine as mch
import longitudinal_tomography.tracking.particles as parts
import longitudinal_tomography.tracking.tracking as tracking
import longitudinal_tomography.tomography.tomography as tomography
import longitudinal_tomography.utils.tomo_output as tomoout
from longitudinal_tomography.utils import tomo_config as conf



# Constant imports
from scipy.constants import c, e, m_p

timing.stop_timing()

# SIMULATION PARAMETERS -------------------------------------------------------
timing.start_timing('blond::set_params')

# Beam parameters
n_particles = int(1e11) # ?
n_macroparticles = int(1e5) # only in BLonD

distribution_exponent = None
bunch_length_fit = 'full'
distribution_type = 'parabolic_line'
max_bunch_length = 0.5476677e-6
bunch_length = max_bunch_length * 0.6   # [s] between 10% and 90% of RF period (0.5476677e-6)
n_bins_arr = np.array([100, 200, 300])                           # Between 50 and 2000
n_bins = 100

# Machine and RF parameters
radius = 25.0                   # for PSB
bending_radius = 8.239          # for PSB
gamma_transition = 4.1          # for PSB
C = 2 * np.pi * radius          # [m]

# Tracking details
n_turns_arr = np.array([100, 300, 500]) # n_profiles!!!
n_turns = 100
d_turns_arr = np.array([9, 9, 9])
dturns = 9

# Derived parameters
E_0 = m_p * c**2 / e            # [eV]
E_kin = 2e9                     # [eV] FT: 2e9 (or 1.4e9), FB: 160e6
tot_beam_energy = E_0 + E_kin   # [eV]
sync_momentum = np.sqrt(tot_beam_energy**2 + E_0**2)    # [eV]
momentum_compaction = 1 / gamma_transition**2
charge = 1 # -1 if electron
b0 = sync_momentum / bending_radius / c                 # [T]

gamma = tot_beam_energy / E_0
beta = np.sqrt(1.0-1.0/gamma**2.0)

# Cavity parameters
n_rf_systems = 1
h = 1.0 # for PSB
voltage_program = 24e3 # for PSB
phi_offset = np.pi

timing.stop_timing()
# DEFINE BLonD OBJECTS --------------------------------------------------------

timing.start_timing("set_device")
if os.getenv('SINGLE_PREC') is not None:
    conf.AppConfig.set_single_precision() if os.getenv('SINGLE_PREC') == 'True' else conf.AppConfig.set_double_precision()

if os.getenv('MODE') is not None:
    if os.getenv('MODE') == "CPP":
        conf.AppConfig.use_cpu()
    elif os.getenv('MODE') == "CuPy":
        conf.AppConfig.use_cupy()
        timing.mode = timing.Mode.CUPY
    elif os.getenv('MODE') == "CUDA":
        conf.AppConfig.use_gpu()
        timing.mode = timing.Mode.CUPY
    else:
        print("No mode given, using CPP")
        conf.AppConfig.use_cpu()
timing.stop_timing()

timing.start_timing('blond:create_objects')
general_params = Ring(C, momentum_compaction, sync_momentum, Proton(),
                    n_turns, bending_radius=bending_radius)
RF_st_par = RFStation(general_params, [h], [voltage_program], [phi_offset],
                    n_rf_systems)
beam = Beam(general_params, n_macroparticles, n_particles)
ring_RF_section = RingAndRFTracker(RF_st_par, beam)
full_tracker = FullRingAndRF([ring_RF_section])

fs = RF_st_par.omega_s0[0]/2/np.pi
bucket_length = 2.0 * np.pi / RF_st_par.omega_rf[0,0]

slice_beam = Profile(beam, CutOptions(cut_left=0, cut_right=bucket_length, n_slices=n_bins))
monitor = SlicesMonitor('./blonddata', n_turns, slice_beam)

timing.stop_timing()

# BEAM GENERATION -------------------------------------------------------------

timing.start_timing('blond::match_and_track')

distr = matched_from_distribution_function(beam, full_tracker,
                                distribution_type=distribution_type,
                                distribution_exponent=distribution_exponent,
                                bunch_length=bunch_length,
                                bunch_length_fit=bunch_length_fit,
                                distribution_variable='Action', seed=18)

beam.dE *= 0.3

# BLonD Tracking

bunch_profiles = np.zeros((n_turns, n_bins))

for i in range(n_turns):
    full_tracker.track()
    slice_beam.track()
    bunch_profiles[i] = slice_beam.n_macroparticles
    # BUNCH parameter?
    monitor.track("")
monitor.close()
timing.stop_timing()

#import blond.plots.plot_beams as bpb

#bpb.plot_long_phase_space(general_params, RF_st_par, beam, 0, 547.8e-9, -65e6, 65e6, show_plot = True, separatrix_plot = True, histograms_plot = False)

# TOMOGRAPHY PROCESS ----------------------------------------------------------

# DEFINE MACHINE

timing.start_timing('define_machine_object')

n_profiles = len(bunch_profiles)
dtbin = slice_beam.bin_centers[1] - slice_beam.bin_centers[0]

machine_args = {
    'output_dir':           '/tmp/',
    'dtbin':                dtbin,
    'dturns':               dturns,
    'synch_part_x':         n_bins // 2, #np.argmax(bunch_profiles[0]),
    'demax':                -1.E6,              # noqa
    'filmstart':            0,
    'filmstop':             1,
    'filmstep':             1,
    'niter':                20,
    'snpt':                 4, # Square root of particles pr. cell of phase space.
    'full_pp_flag':         False,
    'beam_ref_frame':       0,
    'machine_ref_frame':    0,
    'vrf1':                 voltage_program,
    'vrf1dot':              0.0,
    'vrf2':                 0.0,
    'vrf2dot':              0.0,
    'h_num':                h,
    'h_ratio':              2.0,
    'phi12':                phi_offset, #0.4007821253666541,
    'b0':                   b0,
    'bdot':                 0.0,
    'mean_orbit_rad':       radius,
    'bending_rad':          bending_radius,
    'trans_gamma':          gamma_transition,
    'rest_energy':          E_0,
    'charge':               charge,
    'self_field_flag':      False,
    'g_coupling':           0.0,
    'zwall_over_n':         0.0,
    'pickup_sensitivity':   0.36,
    'nprofiles':            n_profiles,
    'nbins':                n_bins,
    'min_dt':               0.0,
    'max_dt':               dtbin * n_bins
}
machine = mch.Machine(**machine_args)
timing.stop_timing()

# bunch_profiles = waterfall?
timing.start_timing('create_profile')
waterfall = conf.array(bunch_profiles)
timing.stop_timing()

timing.start_timing('values_at_turns')
machine.values_at_turns()
timing.stop_timing()

timing.start_timing('tracking::create_tracker')
tracker = tracking.Tracking(machine)
reconstruct_idx = machine.filmstart
timing.stop_timing()

xp, yp = tracker.track(reconstruct_idx)

timing.start_timing("physical_to_coords")
xp, yp = parts.physical_to_coords(
    xp, yp, machine, tracker.particles.xorigin,
    tracker.particles.dEbin)
timing.stop_timing()

timing.start_timing("ready_for_tomo")
xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)
timing.stop_timing()

timing.start_timing("cast_coords_to_cpu")
xp = conf.cast_to_cpu(xp)
yp = conf.cast_to_cpu(yp)
timing.stop_timing()

end_time = time.time()

if os.getenv("REPORT_FILENAME") is not None and os.getenv("REPORT_FILENAME") != "":
        report_filename = os.getenv("REPORT_FILENAME")
        timing.report(total_time = (end_time - start_time) * 1e3, out_file=report_filename + f"-baseprog-Autotomo")
        timing.reset()
else:
    timing.report(total_time = (end_time - start_time) * 1e3)
    timing.reset()

precisions = ["single", "double"]

for prec in precisions:
    if prec == "double":
        conf.AppConfig.set_double_precision()
    elif prec == "single":
        conf.AppConfig.set_single_precision()

    it = 100

#for n_bins in n_bins_arr:
#    for n_turns, dturns in zip(n_turns_arr, d_turns_arr):
    firstit = True
    for i in range(it):
        if firstit:
            start_time = time.time()

        timing.start_timing("cast_coords")
        xp1 = conf.cast(xp)
        yp1 = conf.cast(yp)
        timing.stop_timing()

        timing.start_timing("create_tomo_object")
        tomo = tomography.Tomography(waterfall, xp1, yp1)
        timing.stop_timing()
        weight = tomo.run(niter=machine.niter)

        timing.start_timing("create_phase_space")
        t_range, E_range, density = dtreat.phase_space(tomo, machine,
                                                    reconstruct_idx)
        timing.stop_timing()

        timing.start_timing("cast_to_cpu_density")
        density = conf.cast_to_cpu(density)
        timing.stop_timing()

        if firstit:
            end_time = time.time()
            if os.getenv("REPORT_FILENAME") is not None and os.getenv("REPORT_FILENAME") != "":
                report_filename = os.getenv("REPORT_FILENAME")
                timing.report(total_time = (end_time - start_time) * 1e3, out_file=report_filename + f"-{prec}-{n_bins}-bins-{n_turns}-profs-Autotomo-it1")
                timing.reset()
            else:
                timing.report(total_time = (end_time - start_time) * 1e3)
                timing.reset()
            firstit = False
            start_time = time.time()

    end_time = time.time()
    if os.getenv("REPORT_FILENAME") is not None and os.getenv("REPORT_FILENAME") != "":
        report_filename = os.getenv("REPORT_FILENAME")
        timing.report(total_time = (end_time - start_time) * 1e3, out_file=report_filename + f"-{prec}-{n_bins}-bins-{n_turns}-profs-Autotomo-itrest")
        timing.reset()
    else:
        timing.report(total_time = (end_time - start_time) * 1e3)
        timing.reset()