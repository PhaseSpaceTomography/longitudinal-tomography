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


def kick(dphi, dE, charge, vrf1, vrf2, phi0, phi12, h_ratio, acc_kick):
    return dE + charge * (vrf1 * np.sin(dphi + phi0)
                          + vrf2 * np.sin(h_ratio * (dphi + phi0 - phi12))) \
           - acc_kick


ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', 'C500MidPhaseNoise.dat')

file = []
with open(in_file_pth, 'r') as f:
    for i in range(98):
        file.append(f.readline().strip())

machine, _ = tin.txt_input_to_machine(file)

bunch_position = machine.synch_part_x * machine.dtbin * 0.0
bunch_length = machine.nbins * machine.dtbin
bunch_energy = 0.0
energy_spread = 1.0E6
n_parts = int(4E4)

dphi, denergy = generate_bunch(
    bunch_position, bunch_length, bunch_energy,
    energy_spread, n_parts)

dphi = dphi * np.pi / (machine.nbins * machine.dtbin)

nturns = machine.dturns * (machine.nprofiles - 1)

# One frame is stored per measured profile, dturns machine turns apart
all_dphi = [dphi.tolist()]
all_denergy = [denergy.tolist()]
for turn in range(1, nturns + 1):
    dphi = drift(dphi, denergy, machine.h_num, machine.beta0[turn - 1],
                 machine.e0[turn - 1], machine.eta0[turn - 1])

    denergy = kick(dphi, denergy, machine.q, machine.vrf1_at_turn[turn],
                   machine.vrf2_at_turn[turn], machine.phi0[turn],
                   machine.phi12, machine.h_ratio, machine.deltaE0[turn])
    if turn % machine.dturns == 0:
        all_dphi.append(dphi.tolist())
        all_denergy.append(denergy.tolist())
all_dphi = np.array(all_dphi)
all_denergy = np.array(all_denergy)

# Origin and energy bin size of the reconstructed phase space
ref_turn = machine.beam_ref_frame * machine.dturns
xorigin = (machine.phi0[ref_turn]
           / (machine.h_num * machine.omega_rev0[ref_turn] * machine.dtbin)
           - machine.synch_part_x)
dEbin = (machine.beta0[ref_turn]
         * np.sqrt(machine.e0[ref_turn] * machine.q
                   * machine.vrf1_at_turn[ref_turn]
                   * np.cos(machine.phi0[ref_turn])
                   / (2 * np.pi * machine.h_num * machine.eta0[ref_turn]))
         * machine.dtbin * machine.h_num * machine.omega_rev0[ref_turn])

# Converting from phase [rad] and energy [eV] to bins of the phase space
# coordinate system, one profile per row.
turns = np.arange(machine.nprofiles) * machine.dturns
phi0 = machine.phi0[turns].reshape(-1, 1)
omega_rev0 = machine.omega_rev0[turns].reshape(-1, 1)

xp = ((all_dphi + phi0)
      / (machine.h_num * omega_rev0 * machine.dtbin) - xorigin)
yp = all_denergy / dEbin + machine.synch_part_y

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
