# General utils
import time as tm
import sys
import matplotlib.pyplot as plt
import numpy as np

# Tomo modules
from parameters import Parameters
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
parameters = Parameters()
parameters.parse_from_txt(raw_param)
parameters.fill()

output_path = OutputHandler.adjust_outpath(parameters.output_dir)

# Setting up time space object
timespace = TimeSpace(parameters)
timespace.create(raw_data)

# Initiating particles obj.
particles = Particles(timespace)

# ------------------------------------------------------------------------------
# EXAMPLE, use of particles object - manual coordinates
# Setting the initial position of the particles using the xy coordinate system.
# The particles can also be directly tracked, if given to the 
#  tracking object in phase and and energy.
# ------------------------------------------------------------------------------
xcoords = np.array([100, 100, 100, 75, 100, 125])
ycoords = np.array([75, 100, 125, 100, 100, 100])

particles.set_coordinates(xcoords, ycoords)
dphi, deneregy = particles.init_coords_to_physical(turn=0)

tracker = Tracking(parameters)
xp, yp = tracker.track((dphi, deneregy))

# Converting from physical units to coordinate system
xp, yp = particles.physical_to_coords(xp, yp)

# ------------------------------------------------------------------------------
# EXAMPLE, use of particles object - automatic distribution
# ------------------------------------------------------------------------------
particles.homogeneous_distribution(ff=True)
dphi, deneregy = particles.init_coords_to_physical(turn=0)

tracker = Tracking(parameters)
xp, yp = tracker.track((dphi, deneregy))
xp, yp = particles.physical_to_coords(xp, yp)

# Needed for tomo routine.
# Will be removed eventually. 
xp = np.ceil(xp).astype(int).T - 1
yp = np.ceil(yp).astype(int).T - 1

# Tomography!
tomo = TomographyCpp(timespace.profiles, xp)
weight = tomo.run()

# Creating image
nbins = timespace.par.profile_length
rec_prof = 0
image = OutputHandler.create_phase_space_image(
                    xp, yp, weight, nbins, rec_prof)

plt.imshow(image.T, cmap='hot', origin='lower', interpolation='nearest')
plt.show()