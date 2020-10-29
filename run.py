# General utils

# Tomo modules
import tomo.tomography.tomography as tomography
import tomo.tracking.particles as pts
import tomo.tracking.tracking as tracking
import tomo.data.data_treatment as dtreat
import tomo.utils.tomo_input as tomoin
import tomo.utils.tomo_output as tomoout

# =========================
#        Program 
# =========================

# --------------------- FORTRAN SPECIFIC ------------------------

# Loading input
raw_param, raw_data = tomoin.get_user_input()

# Generating machine object
machine, frames = tomoin.txt_input_to_machine(raw_param)
machine.values_at_turns()
waterfall = frames.to_waterfall(raw_data)

# ------------------- END FORTRAN SPESCIFIC -----------------------

# Creating profiles object
profiles = tomoin.raw_data_to_profiles(
                waterfall, machine, frames.rebin, frames.sampling_time)
profiles.calc_profilecharge()

if profiles.machine.synch_part_x < 0:
    fit_info = dtreat.fit_synch_part_x(profiles)
    machine.load_fitted_synch_part_x_ftn(fit_info)
reconstr_idx = machine.filmstart

# Tracking...
tracker = tracking.Tracking(machine)
tracker.enable_fortran_output(profiles.profile_charge)

if tracker.self_field_flag:
    profiles.calc_self_fields()
    tracker.enable_self_fields(profiles)

xp, yp = tracker.track(reconstr_idx)

# Converting from physical coordinates ([rad], [eV])
# to phase space coordinates.
if not tracker.self_field_flag:
    xp, yp = pts.physical_to_coords(
                xp, yp, machine, tracker.particles.xorigin,
                tracker.particles.dEbin)

# Filters out lost particles, transposes particle matrix, casts to np.int32.
xp, yp = pts.ready_for_tomography(xp, yp, machine.nbins)

# Tomography!
tomo = tomography.TomographyCpp(profiles.waterfall, xp)
weight = tomo.run(verbose=True)

# Creating and presenting phase-space image
nbins = profiles.machine.nbins
image = tomoout.create_phase_space_image(xp, yp, weight, nbins, reconstr_idx)

tomoout.show(image, tomo.diff, profiles.waterfall[reconstr_idx])
