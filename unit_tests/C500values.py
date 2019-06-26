import numpy as np

# Class for having easy access to resources during testing.
# All values are gathered from the execution of the input file 'C500MidPhaseNoise.dat'

class C500:
    def __init__(self):
        self.path = r"./resources/C500MidPhaseNoise/"

        self.arrays = {
            "beta0": np.genfromtxt(self.path + "beta0.dat"),
            "data_baseline_subtr": np.genfromtxt(
                                        self.path
                                        + "data_baseline_subtracted.dat"),
            "deltaE0": np.genfromtxt(self.path + "deltaE0.dat"),
            "e0":  np.genfromtxt(self.path + "e0.dat"),
            "eta0": np.genfromtxt(self.path + "eta0.dat"),
            "omegarev0": np.genfromtxt(self.path + "omegarev0.dat"),
            "phi0": np.genfromtxt(self.path + "phi0.dat"),
            "raw_profiles": np.genfromtxt(self.path + "raw_profiles.dat"),
            "rebinned_profiles_even": np.genfromtxt(
                                        self.path
                                        + "rebinned_profiles.dat"),
            "rebinned_profiles_odd": np.genfromtxt(
                                        self.path
                                        + "rebinned_profiles_odd.dat"),
            "profiles": np.genfromtxt(self.path + "profiles.dat"),
            "sfc": np.genfromtxt(self.path + "sfc.dat"),
            "time_at_turn": np.genfromtxt(self.path + "time_at_turn.dat"),
            "dphase": np.genfromtxt(self.path + "dphase.dat"),
            "phases": np.genfromtxt(self.path + "phases.dat"),
            "imin": np.load(self.path + "imin.npy"),
            "imax": np.load(self.path + "imax.npy"),
            "jmin": np.load(self.path + "jmin.npy").reshape(205),
            "jmax": np.load(self.path + "jmax.npy").reshape(205)
        }

        self.values = {
            "frame_skipcount": 0,
            "frame_length": 1000,
            "preskip_length": 130,
            "postskip_length": 50,
            "profile_length_before_rebin": 820,
            "init_profile_length": 820,
            "reb_profile_length": 205,
            "profile_count": 100,
            "rebin": 4,
            "dtbin": 1.9999999999999997e-09,
            "dturns": 12,
            "pickup_sens": 0.36,
            "xat0": 88.00000000000001,
            "yat0": 102.5,
            "xorigin": -69.73326295579088,
            "fit_xat0": 0.0,
            "bdot": 1.882500000000023,
            "beam_ref_frame": 1,
            "phiwrap": 6.283185307179586,
            "wraplength": 368,
            "vrf1": 7945.403672852664,
            "vrf1dot": 0.0,
            "vrf2": -0.0,
            "vrf2dot": 0.0,
            "q": 1.0,
            "h_num": 1.0,
            "hratio": 2.0,
            "debin": 23340.63328895732,
            "demax": -1000000.0,
            "phi12": 0.3116495273194016,
            "tanfoot_up": 0.0,
            "tanfoot_low": 0.0,
            "filmstart": 1,
            "filmstop": 1,
            "filmstep": 1,
            "allbin_min": 2,
            "allbin_max": 203,
            "snpt": 4
        }

        self.directory = r"resources/C500MidPhaseNoise/"

    def get_reconstruction_values(self):
        rec_vals = {
            "points": np.load(self.path + "points.npy"),
            "mapsi": np.load(self.path + "mapsi.npy"),
            "mapsw": np.load(self.path + "mapsw.npy"),
            "maps": np.load(self.path + "maps.npy"),
            "init_xp": np.load(self.path + "init_xp.npy"),
            "init_yp": np.load(self.path + "init_yp.npy"),
            "first_xp": np.load(self.path + "longtrack_first_xp.npy"),
            "first_yp": np.load(self.path + "longtrack_first_yp.npy"),
            "first_rev_xp": np.load(self.path
                                    + "longtrack_first_reversed_xp.npy"),
            "first_rev_yp": np.load(self.path
                                    + "longtrack_first_reversed_yp.npy"),
            "rweights": np.load(self.path + "reversedweights.npy"),
            "needed_maps": 2539200,
            "fmlistlength": 16
        }
        return rec_vals

    def get_tomography_values(self):
        tomo_vals = {
            "first_backproj": np.load(self.path + "first_backproj.npy"),
            "first_dproj": np.load(self.path + "first_diffprofiles.npy"),
            "ps_before_norm": np.load(self.path
                                      + "phase_space0_to_norm.npy"),
            "ps_after_norm": np.load(self.path
                                     + "phase_space0_after_norm.npy")

        }
        return tomo_vals
