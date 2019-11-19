# General utils
import time as tm
import sys
import matplotlib.pyplot as plt
import numpy as np

# Tomo modules
from machine import Machine
from time_space import TimeSpace
from particles import Particles
from tracking.tracking import Tracking
from tomography.tomography_cpp import TomographyCpp
from utils.tomo_io import InputHandler, OutputHandler

# =========================
#        Program 
# =========================

# Loading input
raw_param, raw_data = InputHandler.get_input_from_file()
machine = Machine()
machine.parse_from_txt(raw_param)
machine.fill()

output_path = OutputHandler.adjust_outpath(machine.output_dir)

# Setting up time space object
timespace = TimeSpace(machine)
timespace.create(raw_data)

# Initiating particles obj.
particles = Particles(timespace)

# ------------------------------------------------------------------------------
# EXAMPLE, use of particles object - automatic distribution
# ------------------------------------------------------------------------------

# TEMP
reconstr_idx = timespace.machine.beam_ref_frame - 1
reconstruct_turn = reconstr_idx * timespace.machine.dturns
# END TEMP

particles.homogeneous_distribution(ff=True)

# particles.fortran_homogeneous_distribution(timespace)
dphi, deneregy = particles.init_coords_to_physical(turn=reconstruct_turn)

tracker = Tracking(machine)

xp, yp = tracker.track((dphi, deneregy), rec_prof=reconstr_idx)
xp, yp = particles.physical_to_coords(xp, yp)

xp, yp, lost = particles.filter_lost_paricles(xp, yp)
print(f'Lost particles: {lost}')

# Needed for tomo routine.
xp = xp.astype(np.int32).T
yp = yp.astype(np.int32).T

# Tomography!
tomo = TomographyCpp(timespace.profiles, xp)
weight = tomo.run()

# Creating image
nbins = timespace.machine.nbins
image = OutputHandler.create_phase_space_image(xp, yp, weight, nbins,
                                               film=reconstr_idx)

OutputHandler.show(image, tomo.diff, timespace.profiles[reconstr_idx])
