import logging as log
import os
import time as tm

import matplotlib.pyplot as plt
import numpy as np

import longitudinal_tomography.exceptions as expt
import longitudinal_tomography.tomography.tomography as tomography
import longitudinal_tomography.tracking.particles as parts
import longitudinal_tomography.tracking.tracking as tracking
import longitudinal_tomography.utils.tomo_data as tdata

import longitudinal_tomography
longitudinal_tomography.use_gpu()

file_name = 'C550MidPhaseNoise.dat'
ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', file_name)

tomo_data = tdata.TomographyData.from_file(in_file_pth)

# Builds the machine, treats the raw data and fits synch_part_x if needed
profiles = tomo_data.to_profiles()
machine = profiles.machine

tomo = tomography.TomographyCpp(profiles.waterfall)

snpt0 = 1
snpt1 = 6
dsnpt = 1
snpts = np.arange(snpt0, snpt1, dsnpt)

diffs = []
dtimes = []
successful_inputs = []
for snpt in snpts:
    print(f'Running tomo using {snpt ** 2} particles per cell of phase space.')
    machine.snpt = snpt

    try:
        t0 = tm.perf_counter()

        tracker = tracking.Tracking(machine)
        xp, yp = tracker.track(machine.filmstart)

        xp, yp = parts.physical_to_coords(
            xp, yp, machine, tracker.particles.xorigin,
            tracker.particles.dEbin)
        xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)

        tomo.xp = xp
        weight = tomo.run(niter=machine.niter)
        tomo.xp = None

        t1 = tm.perf_counter()
    except expt.TomoException as e:
        log.warning(f' {file_name} could not be run with '
                    f'input parameter snpt = {snpt}.\n'
                    f'Error message: {e}')
    else:
        dtimes.append(t1 - t0)
        diffs.append(tomo.diff[-1])
        successful_inputs.append(snpt)

# Plotting
if len(successful_inputs) > 0:
    for value, discr, time in zip(successful_inputs, diffs, dtimes):
        plt.scatter(time, discr, label=f'{value}')

    offset = max(diffs) * 0.05
    ylim_up = max(diffs) + offset
    ylim_low = min(diffs) - offset
    plt.ylim([ylim_low, ylim_up])

    ax = plt.gca()
    ax.legend(title='snpt')
    ax.set_xlabel('Execution time [s]')
    ax.set_ylabel('Discrepancy')
    ax.ticklabel_format(axis='y', scilimits=(0, 0), style='sci')
    plt.show()
else:
    print('No runs were successful.')
