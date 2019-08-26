import unittest
import numpy as np
import numpy.testing as nptest
from tomo.map_info import MapInfo
from tomo.time_space import TimeSpace
from tomo.parameters import Parameters
from unit_tests.C500values import C500


class TestMap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Getting correct values and arrays for C500MidPhaseNoise input
        cls.c500 = C500()
        # Making MapInfo object for calling functions
        cls.mi = MapInfo.__new__(MapInfo)

    def test_energy_binning_C500(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays
        turn_now = (cv["beam_ref_frame"] - 1) * cv["dturns"]

        debin = MapInfo._energy_binning(
                                MapInfo, cv["vrf1"], cv["vrf1dot"],
                                cv["vrf2"], cv["vrf2dot"], cv["yat0"],
                                ca["dphase"], cv["reb_profile_length"],
                                cv["q"], ca["e0"], ca["phi0"],
                                cv["h_num"], ca["eta0"], cv["dtbin"],
                                ca["omegarev0"], cv["demax"], ca["beta0"],
                                cv["hratio"], cv["phi12"],
                                ca["time_at_turn"], ca["phases"], turn_now)

        self.assertAlmostEqual(debin, cv["debin"],
                               msg="dEbin was not correctly calculated")

    def test_calc_phases_C500(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays
        turn = 0
        indarr = np.arange(cv["reb_profile_length"] + 1)

        phases = MapInfo.calculate_phases_turn(
                            MapInfo, cv["xorigin"], cv["dtbin"],
                            cv["h_num"], ca["omegarev0"][turn],
                            cv["reb_profile_length"], indarr)

        nptest.assert_almost_equal(phases, ca["phases"],
                                   err_msg="Difference in calculation of phases")

    def test_trajectory_height_arr_C500(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays
        energy = 0.0
        turn = 0
        test_ans = 287470.8666145134

        ans = MapInfo.trajectoryheight(
                        ca["phases"], ca["phases"][0],
                        energy, cv["q"], ca["dphase"], ca["phi0"],
                        cv["vrf1"], cv["vrf1dot"], cv["vrf2"], cv["vrf2dot"],
                        cv["hratio"], cv["phi12"], ca["time_at_turn"],
                        turn)

        self.assertAlmostEqual(ans[0], 0.0,
                               msg="trajectory height (array): "
                                   "first element not correct")
        self.assertAlmostEqual(ans[1], test_ans,
                               msg="trajectory height (array): "
                                   "second element not correct")
        self.assertEqual(len(ca["phases"]), len(ans),
                         msg="trajectory height (array): "
                             "length of array not correct")

    def test_trajectory_height_val_C500(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays
        energy = 0.0
        turn = 0
        expected_ans = 287470.8666145134

        ans = MapInfo.trajectoryheight(
                        ca["phases"][1], ca["phases"][0],
                        energy, cv["q"], ca["dphase"], ca["phi0"],
                        cv["vrf1"], cv["vrf1dot"], cv["vrf2"], cv["vrf2dot"],
                        cv["hratio"], cv["phi12"], ca["time_at_turn"],
                        turn)

        self.assertAlmostEqual(ans, expected_ans,
                               msg="trajectory height not correct")

    def test_track_all_limits(self):
        cv = TestMap.c500.values

        (jmin, jmax,
         allbin_min,
         allbin_max) = MapInfo._limits_track_all_pxl(
                            MapInfo, cv["reb_profile_length"], cv["yat0"])

        self.assertEqual(allbin_min, [0], msg="Error in allbin_min "
                                              "(track all pxl)")
        self.assertEqual(allbin_max, [205], msg="Error in allbin_max "
                                                "(track all pxl)")
        nptest.assert_equal(jmin, np.full(205, 1, dtype=int),
                            err_msg="Error in calculation of jmin "
                                    "(track all pxl)")
        nptest.assert_equal(jmax, np.full(205, 205, dtype=int),
                            err_msg="Error in calculation of jmax "
                                    "(track all pxl)")

    def test_track_active_pxl(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays
        indarr = np.arange(cv["reb_profile_length"] + 1)

        (jmin,
         jmax,
         allbin_min,
         allbin_max) = TestMap.mi._limits_track_active_pxl(
                                cv["filmstart"], cv["dturns"],
                                cv["reb_profile_length"], indarr,
                                cv["debin"], cv["xorigin"],
                                cv["dtbin"], ca["omegarev0"], cv["h_num"],
                                cv["yat0"], cv["q"], ca["dphase"], ca["phi0"],
                                cv["vrf1"], cv["vrf1dot"], cv["vrf2"],
                                cv["vrf2dot"], cv["hratio"], cv["phi12"],
                                ca["time_at_turn"])

        nptest.assert_equal(jmin, ca["jmin"].flatten(),
                            err_msg="Error in calculation of jmin")
        nptest.assert_equal(jmax, ca["jmax"].flatten(),
                            err_msg="Error in calculation of jmax")
        self.assertEqual(allbin_min, cv["allbin_min"],
                         msg="Error in calculation of allbin_min")
        self.assertEqual(allbin_max, cv["allbin_max"],
                         msg="Error in calculation of allbin_max")

    def test_find_jmax(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays
        turn = 0

        jmax = MapInfo._find_jmax(
                    MapInfo, cv["reb_profile_length"],
                    cv["yat0"], cv["q"], ca["dphase"], ca["phi0"],
                    cv["vrf1"], cv["vrf1dot"],
                    cv["vrf2"], cv["vrf2dot"],
                    cv["hratio"], cv["phi12"],
                    ca["time_at_turn"], ca["phases"],
                    turn, cv["debin"])

        nptest.assert_equal(jmax, ca["jmax"],
                            err_msg="find jmax function failed")

    def test_find_jmin(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays

        jmin = MapInfo._find_jmin(MapInfo,
                                  cv["yat0"],
                                  ca["jmax"])

        nptest.assert_equal(jmin, ca['jmin'],
                            err_msg="find jmin function failed")

    def test_find_allbin_min(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays

        allbin_min = MapInfo._find_allbin_min(
                            MapInfo, ca["jmin"], ca["jmax"],
                            cv["reb_profile_length"])

        nptest.assert_equal(allbin_min, cv["allbin_min"],
                            err_msg="Find allbin_min function failed")

    def test_find_allbin_max(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays

        allbin_max = MapInfo._find_allbin_max(
                        MapInfo, ca["jmin"], ca["jmax"],
                        cv["reb_profile_length"])

        nptest.assert_equal(allbin_max, cv["allbin_max"],
                            err_msg="Find allbin_max function failed")

    def test_bad_ilimits(self):
        ts = TimeSpace.__new__(TimeSpace)
        ts.par = Parameters.__new__(Parameters)

        ts.par.filmstart = self.c500.values['filmstart']
        ts.par.filmstop = self.c500.values['filmstop']
        ts.par.filmstep = self.c500.values['filmstep']
        ts.par.profile_length = self.c500.values['reb_profile_length']

        mi = MapInfo.__new__(MapInfo)
        mi.jmin = np.array([self.c500.arrays['jmin']])
        mi.jmax = np.array([self.c500.arrays['jmax']])

        # Too low imin
        mi.imin = np.array([-1])
        mi.imax = self.c500.arrays['imax']
        with self.assertRaises(Exception):
            mi._assert_correct_arrays(ts)

        # Too low imax
        mi.imin = self.c500.arrays['imin']
        mi.imax = np.array([1])
        with self.assertRaises(Exception):
            mi._assert_correct_arrays(ts)

        mi.imin = self.c500.arrays['imin']
        mi.imax = np.array([self.c500.arrays['jmax'].size + 1])
        with self.assertRaises(Exception):
            mi._assert_correct_arrays(ts)

    def test_bad_jlimits(self):
        ts = TimeSpace.__new__(TimeSpace)
        ts.par = Parameters.__new__(Parameters)

        ts.par.filmstart = self.c500.values['filmstart']
        ts.par.filmstop = self.c500.values['filmstop']
        ts.par.filmstep = self.c500.values['filmstep']
        ts.par.profile_length = self.c500.values['reb_profile_length']

        mi = MapInfo.__new__(MapInfo)
        mi.jmin = np.array([self.c500.arrays['jmin']])
        mi.jmax = np.array([self.c500.arrays['jmax']])
        mi.imin = self.c500.arrays['imin']
        mi.imax = self.c500.arrays['imax']

        # Too low jmin
        mi.jmin[0, 100] = -3
        with self.assertRaises(Exception):
            mi._assert_correct_arrays(ts)
        mi.jmin = np.array([self.c500.arrays['jmin']])

        # Too low jmax
        mi.jmax[0, 5] = 50
        with self.assertRaises(Exception):
            mi._assert_correct_arrays(ts)
        mi.jmax = np.array([self.c500.arrays['jmax']])

        # Too high jmax
        mi.jmax[0, 177] = ts.par.profile_length + 1
        with self.assertRaises(Exception):
            mi._assert_correct_arrays(ts)


if __name__ == '__main__':
    # unittest.main()
    unittest.test_track_active_pxl()

