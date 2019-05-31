import logging
import time as tm
from Time_space import TimeSpace
from MapInfo import MapInfo
from Reconstruct import Reconstruct
from reconstruct_c import Creconstruct
from Tomography import Tomography
logging.basicConfig(level=logging.DEBUG)
PARAMETER_FILE = r"../tomo_action/input_v2.dat"
WORKING_DIR = r"../tomo_action/tmp/"

ts = TimeSpace(PARAMETER_FILE)
ts.save_profiles_text(ts.profiles, WORKING_DIR, "py_profiles.dat")
if ts.par.self_field_flag:
    ts.save_profiles_text(ts.vself[:, :ts.par.profile_length],
                          WORKING_DIR, "py_vself.dat")

mi = MapInfo(ts)
mi.write_jmax_tofile(ts, mi, WORKING_DIR)
mi.write_plotinfo_tofile(ts, mi, WORKING_DIR)

# rec = Reconstruct(ts, mi)
rec = Creconstruct(ts, mi)
t0 = tm.time()
rec.reconstruct()
print("full reconstruction time: " + str(tm.time() - t0))

tomo = Tomography(rec)
for film in range(rec.timespace.par.filmstart - 1,
                  rec.timespace.par.filmstop,
                  rec.timespace.par.filmstep):
    # Creating picture
    tomo.darray, tomo.picture = tomo.run(film)

    # Writing info about discrepancy
    tomo.out_darray_txtfile(WORKING_DIR, film)
    # Writing picture to
    tomo.out_picture(WORKING_DIR, film)

del tomo.darray, tomo.picture, tomo.ts, tomo.mi

