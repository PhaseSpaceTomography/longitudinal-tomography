import os

import longitudinal_tomography.tomography.tomography as tomography
import longitudinal_tomography.tracking.particles as parts
import longitudinal_tomography.tracking.tracking as tracking
import longitudinal_tomography.utils.tomo_data as tdata
import longitudinal_tomography.utils.tomo_output as tomoout

ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', 'C500MidPhaseNoise.dat')

tomo_data = tdata.TomographyData.from_file(in_file_pth)

# Builds the machine, treats the raw data and fits synch_part_x if needed
profiles = tomo_data.to_profiles()
machine = profiles.machine

tracker = tracking.Tracking(machine)
tracker.enable_fortran_output(profiles.profile_charge)

# For including self fields during tracking
# FOR SELF FIELDS
if machine.self_field_flag:
    profiles.calc_self_fields()
    tracker.enable_self_fields(profiles)

for film in range(machine.filmstart, machine.filmstop + 1, machine.filmstep):

    xp, yp = tracker.track(film)

    # Converting from physical coordinates ([rad], [eV])
    # to phase space coordinates.
    if not tracker.self_field_flag:
        xp, yp = parts.physical_to_coords(
            xp, yp, machine, tracker.particles.xorigin,
            tracker.particles.dEbin)

    # Filters out lost particles, transposes particle matrix, casts to np.int32
    xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)

    # Reconstructing phase space
    tomo = tomography.TomographyCpp(profiles.waterfall, xp)
    weight = tomo.run(niter=machine.niter, verbose=True)

    # Creating image for fortran style presentation of phase space.
    image = tomoout.create_phase_space_image(
        xp, yp, weight, machine.nbins, film)
    tomoout.show(image, tomo.diff, profiles.waterfall[film])
