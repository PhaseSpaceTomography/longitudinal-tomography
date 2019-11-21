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
from utils.tomo_io import OutputHandler
import utils.tomo_input as tin

# =========================
#        Program 
# =========================

# Loading input
raw_param, raw_data = tin.get_user_input()

machine = tin.input_to_machine(raw_param)
machine.fill()

output_path = OutputHandler.adjust_outpath(machine.output_dir)

# Setting up time space object
timespace = TimeSpace(machine)
timespace.create(raw_data)

# ------------------------------------------------------------------------------
# EXAMPLE, use of particles object - automatic distribution
# ------------------------------------------------------------------------------

# TEMP
reconstr_idx = machine.beam_ref_frame - 1
reconstruct_turn = reconstr_idx * machine.dturns
# END TEMP

tracker = Tracking(machine)
xp, yp = tracker.track(rec_prof=reconstr_idx)

# Tomography!
tomo = TomographyCpp(timespace.profiles, xp)
weight = tomo.run()

# Creating image
nbins = timespace.machine.nbins
image = OutputHandler.create_phase_space_image(xp, yp, weight, nbins,
                                               film=reconstr_idx)

OutputHandler.show(image, tomo.diff, timespace.profiles[reconstr_idx])
