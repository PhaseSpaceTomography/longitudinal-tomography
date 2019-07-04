import logging
import time as tm
from time_space import TimeSpace
from map_info import MapInfo
from reconstruct_c import ReconstructCpp    # c++ enhanced reconstruction
from reconstruct_py import Reconstruct         # old reconstruction
from tomography import Tomography
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

# rec = ReconstructCpp(ts, mi)
rec = Reconstruct(ts, mi)
tomo = Tomography(rec)

for film in range(rec.timespace.par.filmstart - 1,
                  rec.timespace.par.filmstop,
                  rec.timespace.par.filmstep):

    t0 = tm.perf_counter()
    rec.new_run(film)
    # rec.run(film)
    print(f'total reconstruction time: {str(tm.perf_counter() - t0)}')
    raise SystemExit
    # Creating picture
    tomo.darray, tomo.picture = tomo.run(film)

    # Writing discrepancy history of tomography,
    # and finished picture to file.
    tomo.out_darray_txtfile(WORKING_DIR, film)
    tomo.out_picture(WORKING_DIR, film)

del tomo.darray, tomo.picture, tomo.ts, tomo.mi

