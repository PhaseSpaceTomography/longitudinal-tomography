import logging
from Time_space import TimeSpace
from MapInfo import MapInfo
from Reconstruct import Reconstruct
from Tomography import Tomography

logging.basicConfig(level=logging.DEBUG)
PARAMETER_FILE = r"/home/cgrindhe/tomo_action/input_v2.dat"
WORKING_DIR = r"/home/cgrindhe/tomo_action/tmp/"
# PARAMETER_FILE = r"tmp/input.dat"
# WORKING_DIR = r"tmp/out/"

ts = TimeSpace(PARAMETER_FILE)
ts.save_profiles_text(ts.profiles, WORKING_DIR, "py_profiles.dat")
if ts.par.self_field_flag:
    ts.save_profiles_text(ts.vself[:, :ts.par.profile_length],
                          WORKING_DIR,
                          "py_vself.dat")

mi = MapInfo(ts)
mi.write_jmax_tofile(ts, mi, WORKING_DIR)
mi.write_plotinfo_tofile(ts, mi, WORKING_DIR)

rec = Reconstruct(ts, mi)
rec.reconstruct(rec.timespace, rec.mapinfo)

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

