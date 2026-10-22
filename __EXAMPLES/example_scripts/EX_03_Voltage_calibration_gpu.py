import os

import matplotlib.pyplot as plt
import numpy as np

import longitudinal_tomography.tomography.tomography as tomography
import longitudinal_tomography.tracking.particles as parts
import longitudinal_tomography.tracking.tracking as tracking
import longitudinal_tomography.utils.tomo_data as tdata

import longitudinal_tomography
longitudinal_tomography.use_gpu()
longitudinal_tomography.set_double_precision()

if os.getenv("DATAFILE") is not None:
    datafile = os.getenv("DATAFILE")
else:
    datafile = "flatTopINDIVRotate2.dat"


ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, "input_files", datafile)

tomo_data = tdata.TomographyData.from_file(in_file_pth)

# Builds the machine, treats the raw data and fits synch_part_x if needed
profiles = tomo_data.to_profiles()
machine = profiles.machine

ntest_points = 20
vary_rfv = 1000  # volts
rfv_start = machine.vrf1 - vary_rfv
rfv_end = machine.vrf1 + vary_rfv
rfv_inputs = np.linspace(rfv_start, rfv_end, ntest_points)

tomo = tomography.Tomography(profiles.waterfall)


diffs = []
for rfv in rfv_inputs:
    print(f"Reconstructing using rf-voltage of {rfv:.3f} volt")
    machine.vrf1 = rfv
    machine.values_at_turns()

    tracker = tracking.Tracking(machine)
    xp, yp = tracker.track(machine.filmstart)

    xp, yp = parts.physical_to_coords(
        xp, yp, machine, tracker.particles.xorigin, tracker.particles.dEbin
    )
    xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)

    tomo.xp = xp
    weight = tomo.run(niter=machine.niter)
    tomo.xp = None

    diffs.append(tomo.diff[-1])

plt.plot(rfv_inputs, diffs)
ax = plt.gca()
ax.set_xlabel("Input voltage")
ax.set_ylabel("Discrepancy")
ax.ticklabel_format(axis="y", scilimits=(0, 0), style="sci")
plt.show()
