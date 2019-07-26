import logging
import time as tm
import sys
import matplotlib.pyplot as plt
import numpy as np
from tracking import Tracking
from time_space import TimeSpace
from map_info import MapInfo
from new_tomo_cpp import NewTomographyC

logging.basicConfig(level=logging.DEBUG)

live = False
if live:
    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    except IndexError:
        print('Error: You must provide an input file and an output directory')
        print('Usage: main_testing <input_path> <output_path>')
        sys.exit('Program exit..')
else:
    input_path = '/home/cgrindhe/tomo_v3/tomo_action/input_v2.dat'
    output_path = '/home/cgrindhe/temp/testing_new_longtrack/output/C500_4pxl'

# Making sure that output path ends on a dash
if output_path[-1] != '/':
    output_path += '/'

# Collecting time space parameters and data
ts = TimeSpace(input_path)
ts.save_profiles_text(ts.profiles, output_path, "py_profiles.dat")

if ts.par.self_field_flag:
    ts.save_profiles_text(ts.vself[:, :ts.par.profile_length],
                          output_path, "py_vself.dat")

# Creating map outlining for reconstruction
mi = MapInfo(ts)
mi.write_jmax_tofile(ts, mi, output_path)
mi.write_plotinfo_tofile(ts, mi, output_path)

# Particle tracking
t0 = tm.perf_counter()
tr = Tracking(ts, mi)
xp, yp = tr.track()
print(f'time - tracking: {tm.perf_counter() - t0}s')

# Transposing needed for tomography routine
xp = xp.T
yp = yp.T

# Reconstructing phase space
t0 = tm.perf_counter()
tomo = NewTomographyC(ts, xp, yp)
weight = tomo.run_cpp()
print(f'time - reconstructing phase space: {tm.perf_counter() - t0}s')

# Printing reconstructed phase-space to screen
show = True
if show:
    plt.scatter(xp[:, 0], yp[:, 0], c=weight)
    plt.show()

# Saving to files
print(f'Saving output to directory: {output_path}')
np.save(output_path + 'weight', weight)
np.save(output_path + 'xp', xp)
np.save(output_path + 'yp', yp)
np.save(output_path + 'diff', tomo.diff)
print('Saving complete!')
print('Program finished.')

# The remaining code is kept for reference in case comparisons etc. are needed
# for film in range(ts.par.filmstart - 1,
#                   ts.par.filmstop,
#                   ts.par.filmstep):
#
#     t0 = tm.perf_counter()
#     # rec.new_run(film)
#     # rec.run(film)
#     xp, yp = rec.run_only_particle_track(film)
#
#
#     print(f'total reconstruction time: {str(tm.perf_counter() - t0)}')
#
#     # TOMO
#     tomo = NewTomography(ts, xp, yp)
#     tid = tm.perf_counter()
#     tomo.run4()
#
#     print('tomo time: ' + str(tm.perf_counter() - tid))
#     # tomo.run()
#
#     raise SystemExit
#     # Creating picture
#     tomo.darray, tomo.picture = tomo.run(film)
#
#     # Writing discrepancy history of tomography,
#     # and finished picture to file.
#     tomo.out_darray_txtfile(WORKING_DIR, film)
#     tomo.out_picture(WORKING_DIR, film)
#
# del tomo.darray, tomo.picture, tomo.ts, tomo.mi
