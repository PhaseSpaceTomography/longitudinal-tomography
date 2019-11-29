# General utils
import time as tm
import sys
import matplotlib.pyplot as plt
import numpy as np

# Tomo modules
from tracking.tracking import Tracking
from tomography.tomography_cpp import TomographyCpp
from fit import fit_xat0
from utils.tomo_input import (get_user_input, txt_input_to_machine,
                              raw_data_to_profiles)
from utils.tomo_output import (show, adjust_outpath,
                               create_phase_space_image)

# =========================
#        Program 
# =========================

# --------------------- FORTRAN SPESCIFIC ------------------------

# Loading input
raw_param, raw_data = get_user_input()

# Generating machine object
machine, frames = txt_input_to_machine(raw_param)
machine.values_at_turns()
waterfall = frames.to_waterfall(raw_data)

# ------------------- END FORTRAN SPESCIFIC -----------------------

# Creating profiles object
profiles = raw_data_to_profiles(waterfall, machine, frames.rebin,
                                frames.sampling_time)
profiles.calc_profilecharge()
profiles.calc_self_fields()

if profiles.machine.xat0 < 0:
    fit_info = fit_xat0(profiles)
    machine.load_fitted_xat0_ftn(fit_info)

reconstr_idx = machine.filmstart

# Tracking...
tracker = Tracking(machine)
tracker.enable_fortran_output(profiles.profile_charge)
tracker.enable_self_fields(profiles)

xp, yp = tracker.track(rec_prof=reconstr_idx)

# Tomography!
tomo = TomographyCpp(profiles.waterfall, xp)
weight = tomo.run()

# Creating and presenting phase-space image
nbins = profiles.machine.nbins
image = create_phase_space_image(xp, yp, weight, nbins, reconstr_idx)

show(image, tomo.diff, profiles.waterfall[reconstr_idx])
