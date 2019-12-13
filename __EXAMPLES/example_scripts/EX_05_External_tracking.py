import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import constants
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

def drift(dphi, dE, hnum, beta, E0, eta):
    return dphi - 2 * np.pi * hnum * eta * dE / (beta**2 * E0)

def kick(dphi, dE, charge, voltage, E0):
    return dE + charge * voltage * np.sin(dphi) - E0

# Loading measured data
ex_dir = os.path.realpath(os.path.dirname(__file__)).split('/')[:-1]
in_file_pth = '/'.join(ex_dir + ['/input_files/C500MidPhaseNoise.dat'])

waterfall = np.genfromtxt(in_file_pth, skip_header=98)

sampling_time = 4.999999999999999E-10
nframes = 100
dturns = 12
nturns = nframes * dturns
nbins = int(len(waterfall) / nframes)

waterfall = waterfall.reshape(nframes, nbins)

# plt.imshow(waterfall, cmap='terrain', origin='lower') # <- terrain var fin
# plt.show()

bunch_position = 0.25591559666284924    # [rad]
bunch_length = 1.7210323875576607       # [rad]
bunch_energy = 0.0                      # [eV]
energy_spread = 1969365.9337549524      # [eV]
nparts = int(1E5)

dphi, denergy = generateBunch(bunch_position, bunch_length,
                              bunch_energy, energy_spread, nparts)

energy = 1.33501286E09              # [eV]
energy_kick = 2.4362921438217163E03 # [eV]
beta=0.7113687870661543     
charge=1
voltage=7945.403672852664           # [V]
harmonic=1
eta=0.4344660490259821

# plt.scatter(dphi, denergy)
# plt.show()
all_dphi = np.zeros((nframes, nparts))
all_denergy = np.zeros((nframes, nparts))

print('Tracking...')
for i in range(nturns):
    dphi = drift(dphi, denergy, harmonic, beta, energy, eta)
    denergy = kick(dphi, denergy, charge, voltage, energy_kick)
    if i % dturns == 0:
        all_dphi[int(i / dturns)] = np.copy(dphi)
        all_denergy[int(i / dturns)] = np.copy(denergy)
print('Tracking complete!')




# Save particles for each time frame
# Convert to phase space coordinates
# tomography(xp, waterfall)
# tomo.run()
# show image
