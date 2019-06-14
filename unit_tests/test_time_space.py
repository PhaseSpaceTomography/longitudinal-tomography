import unittest
import numpy as np
import numpy.testing as nptest
from tomo.Time_space import TimeSpace
from unit_tests.C500values import C500

noiseStruct_path = r"resources/noiseStructure2/"

# All tests marked with C500 takes hold in values
#   from the C500MidPhaseNoise input

class TestTimeSpace(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c500 = C500()

    # Testing raw-data inn
    def test_get_data_and_subtract_baseline_C500(self):
        path = TestTimeSpace.c500.path
        ca = TestTimeSpace.c500.arrays
        cv = TestTimeSpace.c500.values
        comparison_data = ca["data_baseline_subtr"]
        data = TimeSpace.get_indata_txt("pipe",
                                        path + "C500MidPhaseNoise.dat")
        data = TimeSpace.subtract_baseline(data,
                                           cv["frame_skipcount"],
                                           cv["beam_ref_frame"],
                                           cv["frame_length"],
                                           cv["preskip_length"],
                                           cv["init_profile_length"])
        nptest.assert_almost_equal(data, comparison_data,
                                   err_msg="Error in calculation of baseline")

    # def test_subtract_baseline_bad_input_C500(self):
    #     cv = TestTimeSpace.c500.values
    #     with self.assertRaises(AssertionError):
    #         data = []
    #         _ = TimeSpace.subtract_baseline(data,
    #                                         cv["frame_skipcount"],
    #                                         cv["beam_ref_frame"],
    #                                         cv["frame_length"],
    #                                         cv["preskip_length"],
    #                                         cv["init_profile_length"])
    #     self.assertTrue("No data found, unable to calculate baseline",
    #                     msg="Unexpected behaviour when input data = []")

    # Testing conversion from raw-data to finished profile
    def test_rawdata_to_profiles_C500(self):
        ca = TestTimeSpace.c500.arrays
        cv = TestTimeSpace.c500.values
        data = ca["data_baseline_subtr"]

        comparing_profiles = ca["raw_profiles"]
        profiles = TimeSpace.rawdata_to_profiles(
                    data, cv["profile_count"], cv["init_profile_length"],
                    cv["frame_skipcount"], cv["frame_length"],
                    cv["preskip_length"], cv["postskip_length"])
        nptest.assert_almost_equal(comparing_profiles.flatten(),
                                   profiles.flatten(),
                                   err_msg="Error in conversion from raw data"
                                           "to profiles")

    def test_rebinning_C500_even(self):
        ca = TestTimeSpace.c500.arrays
        cv = TestTimeSpace.c500.values

        (binned_profiles,
         binned_profile_length) = TimeSpace.rebin(ca["raw_profiles"],
                                                  cv["rebin"],
                                                  cv["init_profile_length"],
                                                  cv["profile_count"])

        nptest.assert_almost_equal(ca["rebinned_profiles_even"],
                                   binned_profiles,
                                   err_msg="Error in re-binning of profiles "
                                           "(rebin factor even)")
        self.assertEqual(cv["reb_profile_length"], binned_profile_length,
                         msg="Error in length of re-binned profiles "
                             "(rebin factor even)")

    def test_rebinning_C500_odd(self):
        ca = TestTimeSpace.c500.arrays
        cv = TestTimeSpace.c500.values
        comparing_profiles = ca["rebinned_profiles_odd"]
        unbinned_profiles = ca["raw_profiles"]

        rebin = 5
        expected_ans = 164

        (binned_profiles,
         binned_profile_length) = TimeSpace.rebin(unbinned_profiles,
                                                  rebin,
                                                  cv["init_profile_length"],
                                                  cv["profile_count"])

        nptest.assert_almost_equal(comparing_profiles, binned_profiles,
                                   err_msg="Error in re-binning of profiles"
                                           "(rebin factor even)")
        self.assertEqual(binned_profile_length, expected_ans,
                         msg="Error in length of re-binned profiles")

    def test_finished_profile_C500(self):
        ca = TestTimeSpace.c500.arrays
        comparing_profiles = ca["profiles"]
        inn_profiles = ca["rebinned_profiles_even"]

        profiles = TimeSpace.negative_profiles_zero(inn_profiles)
        profiles = TimeSpace.normalize_profiles(profiles)
        np.testing.assert_almost_equal(profiles, comparing_profiles,
                                       err_msg="error in normalisation and"
                                               "suppressing of zeroes"
                                               "in profile")

    # Testing calculation of general TimeSpace variables
    def test_profile_charge_C500(self):
        ca = TestTimeSpace.c500.arrays
        cv = TestTimeSpace.c500.values

        profiles = ca["rebinned_profiles_even"]
        profiles = TimeSpace.negative_profiles_zero(profiles)
        ref_beam = 0
        expected = 1597936374926.2097
        self.assertAlmostEqual(TimeSpace.total_profilecharge(
                                   profiles[ref_beam], cv["dtbin"],
                                   cv["rebin"], cv["pickup_sens"]),
                               expected,
                               msg="Error in calculation of profile charge")

    def test_fit_xat0_C500(self):
        path = TestTimeSpace.c500.path
        cv = TestTimeSpace.c500.values

        ts = TimeSpace(path + "C500MidPhaseNoise.dat")
        self.assertAlmostEqual(ts.par.x_origin, cv["xorigin"],
                               msg="Error in calculation of x_origin")
        self.assertEqual(ts.par.tangentfoot_up, cv["tanfoot_up"],
                         msg="Error in calculation of tangentfoot_up")
        self.assertEqual(ts.par.tangentfoot_low, cv["tanfoot_low"],
                         msg="Error in calculation of tangentfoot_low")
        self.assertAlmostEqual(ts.par.xat0, cv["xat0"],
                               msg="Error in calculation of xat0")
        self.assertEqual(ts.par.fit_xat0, cv["fit_xat0"],
                         msg="Error in calculation of fit_xat0")

    def test_calc_xorigin_C500(self):
        cv = TestTimeSpace.c500.values
        ca = TestTimeSpace.c500.arrays
        xorigin = TimeSpace.calc_xorigin(cv["beam_ref_frame"] - 1,
                                         cv["dturns"],
                                         ca["phi0"],
                                         cv["h_num"],
                                         ca["omegarev0"],
                                         cv["dtbin"],
                                         cv["xat0"])
        self.assertAlmostEqual(xorigin, cv["xorigin"],
                               msg="Error in calculation of xorigin")

    def test_fit_xat0_noiseStructure2(self):
        ts = TimeSpace(noiseStruct_path + "noiseStructure2.dat")
        self.assertAlmostEqual(ts.par.x_origin, -40.80842253769685,
                               msg="Error in calculation of x_origin")
        self.assertAlmostEqual(ts.par.tangentfoot_up, 69.1855852950661,
                               msg="Error in calculation of tangentfoot_up")
        self.assertAlmostEqual(ts.par.tangentfoot_low, 12.514238253440908,
                               msg="Error in calculation of tangentfoot_low")
        self.assertAlmostEqual(ts.par.xat0, 40.843755944677525,
                               msg="Error in calculation of xat0")
        self.assertAlmostEqual(ts.par.fit_xat0, 40.843755944677525,
                               msg="Error in calculation of fit_xat0")
        self.assertAlmostEqual(ts.par.bunch_phaselength, 1.8654164808668838,
                               msg="Error in calculation of bunch_phaselength")

    def test_calc_tangentbin_noiseStructure2(self):
        profiles = np.genfromtxt(noiseStruct_path + "profiles.dat")
        trhld = 0.15 * np.max(profiles[0])
        tan_up, tan_low = TimeSpace._calc_tangentbins(profiles[0],
                                                      profile_length=87,
                                                      threshold=trhld)
        self.assertEqual(tan_low, 15,
                         msg="Error in calculation of lower tangent bin")
        self.assertEqual(tan_up, 66,
                         msg="Error in calculation of upper tangent bin")
        self.assertIsInstance(tan_low, int,
                              msg="Lower tangent bin is not integer")
        self.assertIsInstance(tan_up, int,
                              msg="Upper tangent bin is not integer")

    def test_calc_tangentfeet_noiseStructure2(self):
        profiles = np.genfromtxt(noiseStruct_path + "profiles.dat")
        tan_up, tan_low = TimeSpace._calc_tangentfeet(
                            profiles[0], refprofile_index=0,
                            profile_length=87, threshold_value=0.15)
        self.assertEqual(tan_low, 12.514238253440908,
                         msg="Error in calculation of lower foot tangent")
        self.assertEqual(tan_up, 69.1855852950661,
                         msg="Error in calculation of lower foot tangent")

    def test_find_wraplength_C500(self):
        cv = TestTimeSpace.c500.values
        ca = TestTimeSpace.c500.arrays
        phiwrp, wrp_len = TimeSpace._find_wrap_length(
                            cv["profile_count"], cv["dturns"],
                            cv["dtbin"], cv["h_num"], ca["omegarev0"],
                            cv["reb_profile_length"], cv["bdot"])
        self.assertAlmostEqual(phiwrp, cv["phiwrap"],
                               msg="Error in calculation of phiwrap")
        self.assertEqual(wrp_len, cv["wraplength"],
                         msg="Error in calculation of wraplength")

    def test_find_wraplength_C500_bdot_eq0(self):
        cv = TestTimeSpace.c500.values
        ca = TestTimeSpace.c500.arrays
        phiwrp, wrp_len = TimeSpace._find_wrap_length(
                            cv["profile_count"], cv["dturns"],
                            cv["dtbin"], cv["h_num"], ca["omegarev0"],
                            cv["reb_profile_length"], bdot=0)
        self.assertAlmostEqual(phiwrp, 6.283185307179586,
                               msg="Error in calculation of phiwrap "
                                   "(bdot = 0)")
        self.assertEqual(wrp_len, 369,
                         msg="Error in calculation of wraplength "
                             "(bdot = 0)")

    def test_calc_yat0(self):
        self.assertEqual(TimeSpace._find_yat0(205), 102.5,
                         msg="Error in calculation of yat0")

    def test_filter(self):

        # To be written
        pass

    def test_calc_self_field(self):

        # To be written
        pass

if __name__ == '__main__':
    unittest.main()
