# General utils
import time as tm
import sys
import matplotlib.pyplot as plt
import numpy as np

# Tomo modules
from machine import Machine
from profiles import Profiles
from particles import Particles
from tracking.tracking import Tracking
from tomography.tomography_cpp import TomographyCpp
import utils.tomo_input as tin
from utils.tomo_input import get_user_input, input_to_machine
from utils.tomo_output import (show, adjust_outpath,
                               create_phase_space_image)

# =========================
#        Program 
# =========================

# Loading input
raw_param, raw_data = get_user_input()

machine = input_to_machine(raw_param)
machine.fill()

output_path = adjust_outpath(machine.output_dir)

# Setting up profiles object
profiles = Profiles(machine)
profiles.create(raw_data)

# ------------------------------------------------------------------------------
# EXAMPLE, use of particles object - automatic distribution
# ------------------------------------------------------------------------------

reconstr_idx = machine.filmstart
tracker = Tracking(machine)
tracker.show_fortran_output(profiles)
xp, yp = tracker.track(rec_prof=reconstr_idx)

# Tomography!
tomo = TomographyCpp(profiles.profiles, xp)
weight = tomo.run()

# Creating image
nbins = profiles.machine.nbins
image = create_phase_space_image(xp, yp, weight, nbins,
                                 rec_prof=reconstr_idx)

show(image, tomo.diff, profiles.profiles[reconstr_idx])
