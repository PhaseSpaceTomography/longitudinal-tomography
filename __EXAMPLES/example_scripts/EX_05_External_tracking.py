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
from tomography.tomography_py import TomographyPy
from tracking.tracking import Tracking
from utils.exs_tools import show


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

    # Example on how to insert and track your own particles.
    example_track_particles(tracker)

    # Example on tracking your own particles with and without filtering.
    example_particles_outside_bucket(tracker)
    
    # Example on how to set up your own initial distribution of particles,
    #  then to track them and later to use them in the reconstruction.
    example_track_and_tomo(tracker)


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


def example_track_and_tomo(tracker):
    
    # Creating an initial distribution of test particles.
    # The particles will start both inside and outside of the separatrix.
    x = np.arange(tracker.mapinfo.imin,
                  tracker.mapinfo.imax)
    y = np.arange(np.min(tracker.mapinfo.jmin),
                  np.max(tracker.mapinfo.jmax))

    xx, yy = np.meshgrid(x, y)

    xx = np.ascontiguousarray(xx.flatten().astype(float))
    yy = np.ascontiguousarray(yy.flatten().astype(float))

    # Plotting initial distribution of particles
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.set_title('Initial distribution of test particles')
    ax.scatter(xx, yy, s=0.5, label='Test particles')
    ax.plot(tracker.mapinfo.jmin, color='black', label='Bucket area')
    ax.plot(tracker.mapinfo.jmax, color='black')
    ax.set_xlim((-25, tracker.timespace.par.profile_length + 25))
    ax.set_ylim((-25, tracker.timespace.par.profile_length + 25))
    ax.legend()
    plt.show()
    # ------------------------------------------------------------------------

    # Tracking needed for asserting that no particles are outside of the bucket
    #  area after reconstruction.
    xp, yp = tracker.track(initial_coordinates=(xx, yy),
                           filter_lost=True)

    # Plotting the paths of some of the particles
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.set_title('Paths of some tracked particles.\n'
                 'Many particles are filtered out.')
    skit_pts = 500
    ax.plot(xp[:, ::skit_pts], yp[:, ::skit_pts])
    ax.plot(tracker.mapinfo.jmin, color='black', label='Bucket area')
    ax.plot(tracker.mapinfo.jmax, color='black')
    ax.legend()
    plt.show()
    # ------------------------------------------------------------------------

    # Produces the same output as the original Fortran profgram
    # Now needs to be transposed and subtracted by one to fit python/C++ version.
    # This will be changed in future updates.
    xp = xp.T.astype(int) - 1
    yp = yp.T.astype(int) - 1

    # In order to run the tomography, all you need from the
    #  time space object are:
    #  - the measured profiles
    #  - nr of profiles
    #  - nr of bins
    #  - nr of iterations
    # It is important for the tomography algorithom to work, that
    #  none of the particles are at x values outside of the bins
    #  (larger xp value than profile_length - 1)
    # Automatic handling for this in the tomo routine
    #  will be implemented shortly
    tomo = TomographyPy(tracker.timespace, xp, yp)

    particle_weights = tomo.run()

    rec_prof_idx = 0
    phase_space = tomo.create_phase_space_image(
                    xp, yp, particle_weights,
                    tracker.timespace.par.profile_length,
                    rec_prof_idx)

    show(phase_space, tomo.diff, tracker.timespace.profiles[rec_prof_idx])

    
if __name__ == '__main__':
    main()
