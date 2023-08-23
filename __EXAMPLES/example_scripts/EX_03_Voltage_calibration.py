from pyprof import timing
import time

timing.mode = 'timing'

start_time = time.time()

timing.start_timing('import_packages')

import os

import matplotlib.pyplot as plt
import numpy as np

import longitudinal_tomography.tomography.tomography as tomography
import longitudinal_tomography.tracking.particles as parts
import longitudinal_tomography.tracking.tracking as tracking
import longitudinal_tomography.utils.tomo_input as tomoin

from longitudinal_tomography.utils.execution_mode import Mode

mode = Mode.CUPY

timing.stop_timing()

if mode == Mode.CUPY or mode == Mode.CUDA:
    timing.mode = 'cupy'
    timing.start_timing('import_cupy')
    import cupy as cp
    import longitudinal_tomography.tomography.tomography_cupy as tomography_cupy
    timing.stop_timing()


timing.start_timing('read_input_files')

if os.getenv('DATAFILE') is not None:
    datafile = os.getenv('DATAFILE')
else:
    datafile = "flatTopINDIVRotate2.dat"


ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', datafile)
parameter_lines = 98

input_parameters = []
with open(in_file_pth, 'r') as line:
    for i in range(parameter_lines):
        input_parameters.append(line.readline().strip())

raw_data = np.genfromtxt(in_file_pth, skip_header=98)
timing.stop_timing()
timing.start_timing('create_machine_waterfall')
machine, frames = tomoin.txt_input_to_machine(input_parameters)
measured_waterfall = frames.to_waterfall(raw_data)
timing.stop_timing()

timing.start_timing('create_profiles_tomo')
ntest_points = 20
vary_rfv = 1000  # volts
rfv_start = machine.vrf1 - vary_rfv
rfv_end = machine.vrf1 + vary_rfv
rfv_inputs = np.linspace(rfv_start, rfv_end, ntest_points)

profiles = tomoin.raw_data_to_profiles(
    measured_waterfall, machine,
    frames.rebin, frames.sampling_time)

if mode == Mode.CUPY or mode == Mode.CUDA:
    tomo = tomography_cupy.TomographyCuPy(cp.asarray(profiles.waterfall))
else:
    tomo = tomography.TomographyCpp(profiles.waterfall)

timing.stop_timing()

diffs = []
for rfv in rfv_inputs:
    timing.start_timing("values_at_turns")
    print(f'Reconstructing using rf-voltage of {rfv:.3f} volt')
    machine.vrf1 = rfv
    machine.values_at_turns()
    timing.stop_timing()

    timing.start_timing("tracking::create_tracker")
    tracker = tracking.Tracking(machine)
    timing.stop_timing()
    xp, yp = tracker.track(machine.filmstart, mode=mode)

    timing.start_timing("physical_to_coords")
    xp, yp = parts.physical_to_coords(
        xp, yp, machine, tracker.particles.xorigin,
        tracker.particles.dEbin, mode=mode)
    timing.stop_timing()
    timing.start_timing("ready_for_tomo")
    xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins, mode=mode)
    timing.stop_timing()

    timing.start_timing("set_tomo_xp")
    tomo.xp = xp
    timing.stop_timing()
    weight = tomo.run(niter=machine.niter, mode=mode)
    tomo.xp = None

    diffs.append(tomo.diff[-1])

end_time = time.time()

if os.getenv("REPORT_FILENAME") is not None and os.getenv("REPORT_FILENAME") != "":
    report_filename = os.getenv("REPORT_FILENAME")
    timing.report(total_time = (end_time - start_time) * 1e3, out_file=report_filename)
else:
    timing.report(total_time = (end_time - start_time) * 1e3)

plt.plot(rfv_inputs, diffs)
ax = plt.gca()
ax.set_xlabel('Input voltage')
ax.set_ylabel('Discrepancy')
ax.ticklabel_format(axis='y', scilimits=(0, 0), style='sci')
plt.show()