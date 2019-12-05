import numpy as np

import tomo.fit as fit
import tomo.particles as parts
import tomo.tomography.tomography_cpp as tomography
import tomo.tracking.tracking as tracking
import tomo.utils.tomo_input as tomoin
import tomo.utils.tomo_output as tomoout


input_file_pth = '../input_files/C500MidPhaseNoise.dat'
parameter_lines = 98

input_parameters = []
with open(input_file_pth, 'r') as line:
    for i in range(parameter_lines):
        input_parameters.append(line.readline().strip())

raw_data = np.genfromtxt(input_file_pth, skip_header=98, dtype=np.float32)

# Generating machine object
machine, frames = tomoin.txt_input_to_machine(input_parameters)
machine.values_at_turns()
waterfall = frames.to_waterfall(raw_data)

# Creating profiles object
profiles = tomoin.raw_data_to_profiles(
                waterfall, machine, frames.rebin, frames.sampling_time)
profiles.calc_profilecharge()

if profiles.machine.xat0 < 0:
    fit_info = fit.fit_xat0(profiles)
    machine.load_fitted_xat0_ftn(fit_info)

tracker = tracking.Tracking(machine)

# For including self fields during tracking 
if machine.self_field_flag:
    profiles.calc_self_fields()
    tracker.enable_self_fields(profiles)

# Profile to be reconstructed
reconstruct_idx = machine.filmstart

tracker.enable_fortran_output(profiles.profile_charge)
xp, yp = tracker.track(reconstruct_idx)

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
weight = tomo.run()

# Creating image for fortran style presentation of phase space. 
image = tomoout.create_phase_space_image(
            xp, yp, weight, machine.nbins, reconstruct_idx)
tomoout.show(image, tomo.diff, profiles.waterfall[reconstruct_idx])
