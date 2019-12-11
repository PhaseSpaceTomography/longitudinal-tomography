import numpy as np
import matplotlib.pyplot as plt
import time as tm
import logging as log

import tomo.particles as parts
import tomo.tomography.tomography_cpp as tomography
import tomo.tracking.tracking as tracking
import tomo.utils.tomo_input as tomoin


input_file_pth = '../input_files/C550MidPhaseNoise.dat'
parameter_lines = 98

input_parameters = []
with open(input_file_pth, 'r') as line:
    for i in range(parameter_lines):
        input_parameters.append(line.readline().strip())

raw_data = np.genfromtxt(input_file_pth, skip_header=98)

machine, frames = tomoin.txt_input_to_machine(input_parameters)
machine.values_at_turns()

waterfall = frames.to_waterfall(raw_data)

profiles = tomoin.raw_data_to_profiles(
                waterfall, machine, frames.rebin,
                frames.sampling_time)

snpt0 = 1
snpt1 = 6
dsnpt = 1
snpts = np.arange(snpt0, snpt1, dsnpt)

diffs = []
dtimes = []
succsesfull_inputs = []
for snpt in snpts:
    machine.snpt = snpt
    
    try:
        t0 = tm.perf_counter()

        tracker = tracking.Tracking(machine)
        xp, yp = tracker.track(machine.filmstart)

        xp, yp = parts.physical_to_coords(
                    xp, yp, machine, tracker.particles.xorigin,
                    tracker.particles.dEbin)
        xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)

        tomo = tomography.TomographyCpp(profiles.waterfall, xp)
        weight = tomo.run(niter=machine.niter)

        t1 = tm.perf_counter()
    except Exception as e:
        log.warning(f' {input_file_pth} could not be run with '
                    f'input parameter snpt = {snpt}.\n'
                    f'Error message: {e}')
    else:
        dtimes.append(t1 - t0)
        diffs.append(tomo.diff[-1])
        succsesfull_inputs.append(snpt)

# Plotting
if len(succsesfull_inputs) > 0:
    for value, discr, time in zip(succsesfull_inputs, diffs, dtimes):
        plt.scatter(time, discr, label=f'{value}')
    
    offset = max(diffs) * 0.05
    ylim_up = max(diffs) + offset
    ylim_low = min(diffs) - offset
    plt.ylim([ylim_low, ylim_up])
    
    ax = plt.gca()
    ax.legend(title='snpt')
    ax.set_xlabel('Execution time [s]')
    ax.set_ylabel('Discrepancy')
    ax.ticklabel_format(axis='y', scilimits=(0,0), style='sci')
    plt.show()
else:
    print('No runs were successfull.')
