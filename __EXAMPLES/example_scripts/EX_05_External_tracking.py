import os

import matplotlib.pyplot as plt
import numpy as np

import longitudinal_tomography.utils.tomo_input as tin


def generate_bunch(bunch_position, bunch_length,
                   bunch_energy, energy_spread,
                   n_macroparticles):
    # Generating phase and energy arrays
    phase_array = np.linspace(bunch_position - bunch_length / 2,
                              bunch_position + bunch_length / 2,
                              100)

    energy_array = np.linspace(bunch_energy - energy_spread / 2,
                               bunch_energy + energy_spread / 2,
                               100)

    # Getting Hamiltonian on a grid
    phase_grid, deltaE_grid = np.meshgrid(phase_array, energy_array)

    # Bin sizes
    bin_phase = phase_array[1] - phase_array[0]
    bin_energy = energy_array[1] - energy_array[0]

    # Density grid, isodensity lines
    isodensity = ((phase_grid - bunch_position) / bunch_length * 2) ** 2. + \
                 ((deltaE_grid - bunch_energy) / energy_spread * 2) ** 2.
    density_grid = 1 - isodensity ** 2.
    density_grid[density_grid < 0] = 0
    density_grid /= np.sum(density_grid)

    # Generating particles randomly inside the grid cells according to the
    # provided density_grid
    indexes = np.random.choice(np.arange(0, np.size(density_grid)),
                               n_macroparticles, p=density_grid.flatten())

    # Randomize particles inside each grid cell (uniform distribution)
    particle_phase = np.ascontiguousarray(
        phase_grid.flatten()[indexes] + (np.random.rand(
            n_macroparticles) - 0.5) * bin_phase)
    particle_energy = np.ascontiguousarray(
        deltaE_grid.flatten()[indexes] + (np.random.rand(
            n_macroparticles) - 0.5) * bin_energy)

    return particle_phase, particle_energy


def drift(dphi, dE, hnum, beta, E0, eta):
    return dphi - 2 * np.pi * hnum * eta * dE / (beta ** 2 * E0)


def kick(dphi, dE, charge, voltage, E0):
    return dE + charge * voltage * np.sin(dphi) - E0


ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', 'C500MidPhaseNoise.dat')

file = []
with open(in_file_pth, 'r') as f:
    for i in range(98):
        file.append(f.readline().strip())

machine, frame = tin.txt_input_to_machine(file)
machine.values_at_turns()

bunch_position = machine.synch_part_x * machine.dtbin * 0.0
bunch_length = machine.nbins * machine.dtbin
bunch_energy = 0.0
energy_spread = 1.0E6
n_parts = int(4E4)

dphi, denergy = generate_bunch(
    bunch_position, bunch_length, bunch_energy,
    energy_spread, n_parts)

dphi = dphi * np.pi / (machine.nbins * machine.dtbin)

all_dphi = []
all_denergy = []
for i in range(1000):
    dphi = drift(dphi, denergy, machine.h_num, machine.beta0[0],
                 machine.e_rest, machine.eta0[0])

    denergy = kick(dphi, denergy, machine.q, 1000, 0)
    if i % 10 == 0:
        all_dphi.append(dphi.tolist())
        all_denergy.append(denergy.tolist())
all_dphi = np.array(all_dphi)
all_denergy = np.array(all_denergy)

xorigin = bunch_position - np.min(all_dphi[0])
xp = all_dphi / np.pi * machine.nbins - xorigin

dEmax = np.max(all_denergy)
yp = all_denergy / dEmax * machine.nbins + machine.nbins / 2

for x, y in zip(xp[::10], yp[::10]):
    plt.scatter(x, y, s=0.5)
    plt.show()

# Filter particles
# transpose particles
# do tomography

# xorigin plot
# ------------
# x = np.linspace(-np.pi/2, np.pi/2, 50)
# plt.plot(x, np.sin(x))
# plt.plot([x[0], x[-1]], [0, 0], color='black')
# plt.plot([np.min(all_dphi[0]), np.min(all_dphi[0])], [-1, 1], color='g')
# plt.show()

# Particle trajectories
# ---------------------
# ipts = [0, 20, 200, 2000, 20000]
# plt.plot(all_dphi[:,ipts], all_denergy[:,ipts])
# plt.show()
