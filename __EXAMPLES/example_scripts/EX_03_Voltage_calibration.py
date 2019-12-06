import numpy as np
import matplotlib.pyplot as plt

import tomo.particles as parts
import tomo.tomography.tomography_cpp as tomography
import tomo.tracking.tracking as tracking
import tomo.utils.tomo_input as tomoin


input_file_pth = '../input_files/flatTopINDIVRotate2.dat'
parameter_lines = 98

input_parameters = []
with open(input_file_pth, 'r') as line:
    for i in range(parameter_lines):
        input_parameters.append(line.readline().strip())

raw_data = np.genfromtxt(input_file_pth, skip_header=98)

machine, frames = tomoin.txt_input_to_machine(input_parameters)
waterfall = frames.to_waterfall(raw_data)

ntest_points = 20
vary_rfv = 1000  # in volts
rfv_start = machine.vrf1 - vary_rfv
rfv_end = machine.vrf1 + vary_rfv
rfv_inputs = np.linspace(rfv_start, rfv_end, ntest_points)

profiles = tomoin.raw_data_to_profiles(
                waterfall, machine, frames.rebin,
                frames.sampling_time)

diffs = []
for rfv in rfv_inputs:
    machine.vrf1 = rfv
    machine.values_at_turns()

    tracker = tracking.Tracking(machine)
    xp, yp = tracker.track(machine.filmstart)

    xp, yp = parts.physical_to_coords(
                xp, yp, machine, tracker.particles.xorigin,
                tracker.particles.dEbin)
    xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)

    tomo = tomography.TomographyCpp(profiles.waterfall, xp)
    weight = tomo.run_cpp(niter=machine.niter)

    diffs.append(tomo.diff[-1])

plt.plot(rfv_inputs, diffs)
ax = plt.gca()
ax.set_xlabel('Input voltage')
ax.set_ylabel('Discrepancy')
ax.ticklabel_format(axis='y', scilimits=(0,0), style='sci')
plt.show()
