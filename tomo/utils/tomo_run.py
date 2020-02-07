import sys

# Tomo modules
from ..tracking import tracking as tracking
from ..tomography import tomography_cpp as tomography
from ..utils import data_treatment as dtreat
from ..utils import tomo_input as tomoin
from ..utils import tomo_output as tomoout
from ..tracking import particles as pts

def run_file(file, reconstruct_profile = None):
    
    with open(file, 'r') as file:
        raw_params, raw_data = tomoin._split_input(file.readlines())
    
    machine, frames = tomoin.txt_input_to_machine(raw_params)
    machine.values_at_turns()
    waterfall = frames.to_waterfall(raw_data)
    
    profiles = tomoin.raw_data_to_profiles(waterfall, machine, frames.rebin, 
                                           frames.sampling_time)
    profiles.calc_profilecharge()
    
    if profiles.machine.synch_part_x < 0:
        fit_info = dtreat.fit_synch_part_x(profiles)
        machine.load_fitted_synch_part_x_ftn(fit_info)
    
    if reconstruct_profile is None:
        reconstr_idx = machine.filmstart
    else:
        reconstr_idx = reconstruct_profile
    
    # Tracking...
    tracker = tracking.Tracking(machine)
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
    tomo = tomography.TomographyCpp(profiles.waterfall, xp, yp)
    _ = tomo.run(verbose=False)
    
    return dtreat.phase_space(tomo, machine, profile = reconstr_idx)