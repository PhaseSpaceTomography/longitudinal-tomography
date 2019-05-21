import unittest
import numpy as np
import numpy.testing as nptest
from tomo.Tomography import Tomography
from unit_tests.C500values import C500


class TestTomography(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # cls.C500_arrays = {
        #     "profiles": np.genfromtxt(C500_path + "profiles.dat"),
        #     "maps": np.load(C500_path + "maps.npy"),
        #     "mapsi": np.load(C500_path + "mapsi.npy"),
        #     "mapsw": np.load(C500_path + "mapsw.npy"),
        #     "rev_mapsw":  np.load(C500_path + "reversedweights.npy"),
        #     "imin": np.load(C500_path + "imin.npy"),
        #     "imax": np.load(C500_path + "imax.npy"),
        #     "jmin": np.load(C500_path + "jmin.npy"),
        #     "jmax": np.load(C500_path + "jmax.npy"),
        #     "back_proj0": np.load(C500_path + "first_backproj.npy"),
        #     "diff_prof0": np.load(C500_path + "first_diffprofiles.npy"),
        #     "ps_before_norm": np.load(C500_path
        #                               + "phase_space0_to_norm.npy"),
        #     "ps_after_norm": np.load(C500_path
        #                              + "phase_space0_after_norm.npy")
        # }
        cls.c500 = C500()
        cls.rec_vals = cls.c500.get_reconstruction_values()
        cls.tomo_vals = cls.c500.get_tomography_values()

    def test_backproject(self):
        ca = TestTomography.c500.arrays
        cv = TestTomography.c500.values
        tv = TestTomography.tomo_vals
        rv = TestTomography.rec_vals
        ppath = TestTomography.c500.path + "profiles.dat"
        phase_space = np.zeros((cv["reb_profile_length"],
                                cv["reb_profile_length"]))
        Tomography.backproject(np.genfromtxt(ppath),  # TODO: check out
                               phase_space,
                               ca["imin"][0],
                               ca["imax"][0],
                               ca["jmin"],
                               ca["jmax"],
                               rv["maps"],
                               rv["mapsi"],
                               rv["mapsw"],
                               rv["rweights"],
                               rv["fmlistlength"],
                               cv["profile_count"],
                               cv["snpt"])
        nptest.assert_almost_equal(phase_space, tv["first_backproj"],
                                   err_msg="Error in backprojection")

    def test_project(self):
        ca = TestTomography.c500.arrays
        cv = TestTomography.c500.values
        tv = TestTomography.tomo_vals
        rv = TestTomography.rec_vals
        ppath = TestTomography.c500.path + "profiles.dat"

        diffprofiles = (np.genfromtxt(ppath)  # TODO: check out
                        - Tomography.project(
                            tv["first_backproj"],
                            ca["imin"][0],
                            ca["imax"][0],
                            ca["jmin"],
                            ca["jmax"],
                            rv["maps"],
                            rv["mapsi"],
                            rv["mapsw"],
                            rv["fmlistlength"],
                            cv["snpt"],
                            cv["profile_count"],
                            cv["reb_profile_length"]))
        nptest.assert_almost_equal(diffprofiles, tv["first_dproj"],
                                   err_msg="")

    def test_discrepancy(self):
        cv = TestTomography.c500.values
        tv = TestTomography.tomo_vals

        diff = Tomography.discrepancy(tv["first_dproj"],
                                      cv["reb_profile_length"],
                                      cv["profile_count"])
        self.assertAlmostEqual(diff, 0.0011512802609203203,
                               msg="Error in calculation of "
                                   "discrepancy")

    def test_supress_and_norm(self):
        tv = TestTomography.tomo_vals
        phase_space = tv["ps_before_norm"]

        phase_space = Tomography.supress_zeroes_and_normalize(phase_space)
        nptest.assert_almost_equal(phase_space, tv["ps_after_norm"],
                                   err_msg="Error in suppression and "
                                           "normalization of phase space")



if __name__ == '__main__':
    unittest.main()
