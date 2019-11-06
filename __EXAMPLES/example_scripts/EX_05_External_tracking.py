import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from utils.exs_tools import make_or_clear_dir
sys.path.append('../../tomo')      # Hack
from main import main as tomo_main
from parameters import Parameters
from time_space import TimeSpace
from map_info import MapInfo
from tracking.tracking import Tracking


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

    ts = TimeSpace(param)
    ts.create(raw_data)

    mi = MapInfo(ts)
    dEbin = mi.find_dEbin()

    # For calculating bucket area
    mi.find_ijlimits()

    # Particle tracking
    tracker = Tracking(ts, mi)

    example_track_particles(tracker)
    example_particles_outside_bucket(tracker)


def example_track_particles(tracker):
    
    # Giving start coordinates for five particles
    my_xp = np.array([5, 25, 50, 75, 100])
    my_yp = np.array([100, 100, 100, 100, 100])

    # Tracking the particles through the number of turns
    #  spescified in the C500MidPhaseNoise input.
    # If filter_lost flag is True, particles outside of the bucket is removed.
    xp, yp = tracker.track(initial_coordinates=(my_xp, my_yp),
                           filter_lost=False)
    
    fig, ax = plt.subplots()
    
    # Plot hamiltonians and start coordinates for particles.
    ax.plot(xp, yp, zorder=5)
    ax.scatter(my_xp, my_yp, color='black', marker='.', zorder=10,
               label='Particle\nstart position')
    
    # Showing separatrix
    ax.plot(tracker.mapinfo.jmin, color='black', label='Separatrix', zorder=0)
    ax.plot(tracker.mapinfo.jmax, color='black', zorder=0)
    ax.legend()

    plt.tight_layout()
    plt.show()


def example_particles_outside_bucket(tracker):
    
    my_xp = np.arange(0, 300, 10)
    my_yp = np.zeros(len(my_xp)) + 100

    # Tracking particles outside the bucket area without filtering
    xp_uf, yp_uf = tracker.track(initial_coordinates=(my_xp, my_yp),
                                 filter_lost=False)

    xp_f, yp_f = tracker.track(initial_coordinates=(my_xp, my_yp),
                               filter_lost=True) 
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 13),
                                   sharex=True, sharey=True)

    ax1.set_title('Unfiltered')
    ax1.plot(xp_uf, yp_uf, zorder=5)

    ax2.set_title('Filtered')
    ax2.plot(xp_f, yp_f, zorder=5)

    # Showing separatrix
    for ax in (ax1, ax2):
        ax.plot(tracker.mapinfo.jmin, color='black', label='Separatrix', zorder=0)
        ax.plot(tracker.mapinfo.jmax, color='black', zorder=0)
        ax.legend()

    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()
