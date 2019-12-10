import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import sys


def generateBunch(bunch_position, bunch_length,
                  bunch_energy, energy_spread,
                  n_macroparticles):

    # Generating phase and energy arrays
    phase_array = np.linspace(bunch_position-bunch_length/2,
                              bunch_position+bunch_length/2,
                              100)

    energy_array = np.linspace(bunch_energy-energy_spread/2,
                              bunch_energy+energy_spread/2,
                              100)

    # Getting Hamiltonian on a grid
    phase_grid, deltaE_grid = np.meshgrid(phase_array, energy_array)

    # Bin sizes
    bin_phase = phase_array[1]-phase_array[0]
    bin_energy = energy_array[1]-energy_array[0]

    # Density grid
    isodensity_lines = ((phase_grid-bunch_position)/bunch_length*2)**2. + \
        ((deltaE_grid-bunch_energy)/energy_spread*2)**2.
    density_grid = 1-isodensity_lines**2.
    density_grid[density_grid<0] = 0
    density_grid /= np.sum(density_grid)

    # Generating particles randomly inside the grid cells according to the
    # provided density_grid
    indexes = np.random.choice(np.arange(0,np.size(density_grid)),
                               n_macroparticles, p=density_grid.flatten())   

    # Randomize particles inside each grid cell (uniform distribution)
    particle_phase = (np.ascontiguousarray(phase_grid.flatten()[indexes] +
        (np.random.rand(n_macroparticles) - 0.5) * bin_phase))
    particle_energy = (np.ascontiguousarray(deltaE_grid.flatten()[indexes] +
        (np.random.rand(n_macroparticles) - 0.5) * bin_energy))

    return particle_phase, particle_energy

def kick(dphi, dE, charge, voltage, E0):
    return dE + charge * voltage * np.sin(dphi) - E0


def drift(dphi, dE, hnum, beta, E0, eta):
    return dphi - 2 * np.pi * hnum * eta * dE / (beta**2 * E0)

# Loading measured data
input_file_path = '../input_files/C500MidPhaseNoise.dat'
waterfall = np.genfromtxt('../input_files/C500MidPhaseNoise.dat',
                          skip_header=98)

sampling_time = 4.999999999999999E-10
nframes = 100
dturns = 12
nturns = nframes * dturns
nbins = int(len(waterfall) / nframes)

waterfall = waterfall.reshape(nframes, nbins)

# plt.imshow(waterfall, cmap='terrain', origin='lower') # <- terrain var fin
# plt.show()

bunch_position = 0.0
bunch_length = np.pi
bunch_energy = 0.0
energy_spread = 4E6
nparts = int(1E5)

dphi, denergy = generateBunch(bunch_position, bunch_length,
                              bunch_energy, energy_spread, nparts)

energy = 10E6
beta=0.9
charge=1
voltage=7945.403672852664
harmonic=1
eta=0.01

for i in range(nturns):
    dphi = drift(dphi, denergy, harmonic, beta, energy, eta)
    denergy = kick(dphi, denergy, charge, voltage, energy)
    if i % 24 == 0:
        plt.scatter(dphi, denergy)
        plt.show()

