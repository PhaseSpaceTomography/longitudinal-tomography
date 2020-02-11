import numpy as np
import os

import tomo.utils.data_treatment as dtreat
import tomo.tomography.tomography as tomography
import tomo.tracking.particles as parts
import tomo.tracking.tracking as tracking
import tomo.utils.tomo_input as tomoin
import tomo.utils.tomo_output as tomoout

ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', 'C500MidPhaseNoise.dat')

parameter_lines = 98
input_parameters = []
with open(in_file_pth, 'r') as line:
    for i in range(parameter_lines):
        input_parameters.append(line.readline().strip())

raw_data = np.genfromtxt(in_file_pth, skip_header=98, dtype=np.float32)

# Generating machine object
machine, frames = tomoin.txt_input_to_machine(input_parameters)
machine.values_at_turns()
measured_waterfall = frames.to_waterfall(raw_data)

# Creating profiles object
profiles = tomoin.raw_data_to_profiles(
                measured_waterfall, machine,
                frames.rebin, frames.sampling_time)

profiles.calc_profilecharge()

if profiles.machine.synch_part_x < 0:
    fit_info = dtreat.fit_synch_part_x(profiles)
    machine.load_fitted_synch_part_x_ftn(fit_info)

tracker = tracking.Tracking(machine)
tracker.enable_fortran_output(profiles.profile_charge)

# For including self fields during tracking 
if machine.self_field_flag:
    profiles.calc_self_fields()
    tracker.enable_self_fields(profiles)


for film in range(machine.filmstart, machine.filmstop, machine.filmstep):

    xp, yp = tracker.track(film)

    # Converting from physical coordinates ([rad], [eV])
    # to phase space coordinates.
    if not tracker.self_field_flag:
        xp, yp = parts.physical_to_coords(
                    xp, yp, machine, tracker.particles.xorigin,
                    tracker.particles.dEbin)

    # Filters out lost particles, transposes particle matrix, casts to np.int32.
    xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)

    # Reconstructing phase space
    tomo = tomography.TomographyCpp(profiles.waterfall, xp)
    weight = tomo.run(niter=machine.niter, verbose=True)

    # Creating image for fortran style presentation of phase space. 
    image = tomoout.create_phase_space_image(
                xp, yp, weight, machine.nbins, film)
    tomoout.show(image, tomo.diff, profiles.waterfall[film])
