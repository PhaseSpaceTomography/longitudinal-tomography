import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from utils.exs_tools import show, make_or_clear_dir
sys.path.append('../../tomo')      # Hack
from main import main as tomo_main # Hack
from parameters import Parameters  # Hack
from time_space import TimeSpace
from map_info import MapInfo
from tracking.particle_tracker import ParticleTracker
from cpp_routines.tomolib_wrappers import kick, drift


BASE_PATH = os.path.dirname(
                os.path.realpath(__file__)).split('/')[:-1]
BASE_PATH = '/'.join(BASE_PATH)

INPUT_FILE_DIR = '/'.join([BASE_PATH] + ['input_files'])

def main():
    len_input_param = 98
    input_file = f'{INPUT_FILE_DIR}/C500MidPhaseNoise.dat'
    output_dir = f'{BASE_PATH}/tmp'

    make_or_clear_dir(output_dir)
    
    with open(input_file) as f:
        read = f.readlines()

    # Splitting to parameters and meassured profile data
    raw_parameters = read[:len_input_param]
    raw_data = np.array(read[len_input_param:], dtype=float)
    del(read)

    # Setting output directory
    output_dir_idx = 14
    raw_parameters[output_dir_idx] = f'{output_dir}'

    param = Parameters()
    param.parse_from_txt(raw_parameters)
    param.fill()

    ts = TimeSpace()
    ts.create(param, raw_data)

    mi = MapInfo(ts)
    dEbin = mi.find_dEbin()

    # For calculating bucket area
    mi.find_ijlimits()

    # Particle tracking
    xp, yp = track_few_parts(ts.par, dEbin)
    show_few_parts(ts.par, xp, yp, (mi.jmin, mi.jmax))  


def track_few_parts(par, dEbin):

    # Calculating voltage
    rf1v = (par.vrf1 + par.vrf1dot * par.time_at_turn) * par.q
    rf2v = (par.vrf2 + par.vrf2dot * par.time_at_turn) * par.q

    # Setting initial coordinates for three particles to be tracked
    xp_start = np.array([3.2, 25.0, 100.6, 225.0])
    yp_start = np.array([22.6, 90.8, 66.7, 100.0])

    n_part = len(xp_start)
    n_turns = par.dturns * (par.profile_count - 1)

    # Creating arrays to be filled by tracking routine
    xp = np.zeros((par.profile_count, n_part))
    yp = np.zeros((par.profile_count, n_part))

    # Inserting start values to main array 
    xp[0] = xp_start
    yp[0] = yp_start

    # Go from coordinate system to physical values.
    dphi, denergy = ParticleTracker.coords_to_phase_and_energy(
                                            par, xp[0], yp[0], dEbin)

    # Track particles
    xp, yp = kick_and_drift(par, xp, yp, denergy, dphi,
                            rf1v, rf2v, dEbin, n_turns, n_part)

    return xp, yp


# plots the bucket area, start point and trajectories of particles.
def show_few_parts(par, xp, yp, bucket_area=None):

    fig, ax = plt.subplots()
    
    if bucket_area is not None:
        ax.plot(bucket_area[0], color='black', label='Bucket area')
        ax.plot(bucket_area[1], color='black')

    ax.plot(xp, yp)
    ax.scatter(xp[0,:], yp[0,:], marker='.', color='red')

    rang = (-50, par.profile_length + 50)
    ax.set_xlim(rang)
    ax.set_ylim(rang)
    ax.legend()

    plt.show()


def kick_and_drift(par, xp, yp, denergy, dphi,
                   rf1v, rf2v, dEbin, n_turns, n_part):
    turn = 0
    profile = 0
    print('tracking...')
    while turn < n_turns:
        # Calculating change in phase for each particle at a turn
        dphi = drift(denergy, dphi, par.dphase,
                     n_part, turn)
        turn += 1
        # Calculating change in energy for each particle at a turn
        denergy = kick(par, denergy, dphi, rf1v, rf2v,
                       n_part, turn)
        if turn % par.dturns == 0:
            profile += 1
            xp[profile] = ((dphi + par.phi0[turn])
                           / (float(par.h_num)
                           * par.omega_rev0[turn]
                           * par.dtbin)
                           - par.x_origin)
            yp[profile] = (denergy / float(dEbin)
                           + par.yat0)
            print(f'{profile}')
    return xp, yp

    
if __name__ == '__main__':
    main()
