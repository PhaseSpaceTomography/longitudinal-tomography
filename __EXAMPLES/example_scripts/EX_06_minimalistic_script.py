import os
import sys
import numpy as np
sys.path.append('../../tomo')      # Hack
from tomography.tomography_cpp import TomographyCpp
from utils.exs_tools import show
from utils.tomo_io import InputHandler, OutputHandler

input_path = '/'.join(os.path.realpath(__file__).split('/')[:-2])
input_path += '/input_files'

# Loading finished waterfall and tracked particles
waterfall = np.load(input_path + '/waterfall.npy')
x_coords = np.load(input_path + '/tracked_xp.npy')
y_coords = np.load(input_path + '/tracked_yp.npy')

# Recreating
tomo = TomographyCpp(waterfall, x_coords)
weight = tomo.run_cpp()

# Converting from weights and particles to image
rec_prof = 0
nbins = waterfall.shape[1]
image = OutputHandler.create_phase_space_image(
                    x_coords, y_coords, weight, nbins, rec_prof)

# present output
show(image, tomo.diff, waterfall[rec_prof])
