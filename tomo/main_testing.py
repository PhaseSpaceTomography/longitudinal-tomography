import logging
import time as tm
import sys
from tracking import Tracking
from time_space import TimeSpace
from map_info import MapInfo
# from reconstruct_c import ReconstructCpp    # c++ enhanced reconstruction
# from reconstruct_py import Reconstruct  # old reconstruction
# from tomography import Tomography
from new_tomography import NewTomography

logging.basicConfig(level=logging.DEBUG)
PARAMETER_FILE = r"../tomo_action/input_v2.dat"
WORKING_DIR = r"../tomo_action/tmp/"

# Collecting time space parameters and data
ts = TimeSpace(PARAMETER_FILE)
ts.save_profiles_text(ts.profiles, WORKING_DIR, "py_profiles.dat")

if ts.par.self_field_flag:
    ts.save_profiles_text(ts.vself[:, :ts.par.profile_length],
                          WORKING_DIR, "py_vself.dat")

# Creating map outlining for reconstruction
mi = MapInfo(ts)
mi.write_jmax_tofile(ts, mi, WORKING_DIR)
mi.write_plotinfo_tofile(ts, mi, WORKING_DIR)

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
tomo = NewTomography(ts, xp, yp)
tomo.run4()
print(f'time - reconstructing phase space: {tm.perf_counter() - t0}s')

sys.exit()

# The remaining code is kept for reference in case comparisons etc. are needed
for film in range(ts.par.filmstart - 1,
                  ts.par.filmstop,
                  ts.par.filmstep):

    t0 = tm.perf_counter()
    # rec.new_run(film)
    # rec.run(film)
    xp, yp = rec.run_only_particle_track(film)


    print(f'total reconstruction time: {str(tm.perf_counter() - t0)}')

    # TOMO
    tomo = NewTomography(ts, xp, yp)
    tid = tm.perf_counter()
    tomo.run4()

    print('tomo time: ' + str(tm.perf_counter() - tid))
    # tomo.run()

    raise SystemExit
    # Creating picture
    tomo.darray, tomo.picture = tomo.run(film)

    # Writing discrepancy history of tomography,
    # and finished picture to file.
    tomo.out_darray_txtfile(WORKING_DIR, film)
    tomo.out_picture(WORKING_DIR, film)

del tomo.darray, tomo.picture, tomo.ts, tomo.mi
