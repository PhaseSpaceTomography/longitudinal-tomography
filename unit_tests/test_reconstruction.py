import unittest
import numpy.testing as nptest
import numpy as np
from tomo.reconstruct_py import Reconstruct
from tomo.map_info import MapInfo
from unit_tests.C500values import C500


# Tests for reconstruction class
class TestRec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Getting correct values and arrays for C500MidPhaseNoise input
        cls.c500 = C500()
        cls.rec_vals = cls.c500.get_reconstruction_values()
        # Making MapInfo object for calling functions
        cls.rec = Reconstruct.__new__(Reconstruct)

    @classmethod
    def tearDownClass(cls):
        # Getting correct values and arrays for C500MidPhaseNoise input
        del cls.c500
        del cls.rec_vals
        # Making MapInfo object for calling functions
        del cls.rec

    def test_populate_bins(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        points = TestRec.rec._populate_bins(cv["snpt"])
        nptest.assert_equal(points, rv["points"],
                            err_msg="Initial points calculated incorrectly.")

    def test_number_of_maps(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays
        mi = MapInfo.__new__(MapInfo)
        mi.imin = ca["imin"]
        mi.imax = ca["imax"]
        mi.jmin = np.array([ca["jmin"]])
        mi.jmax = np.array([ca["jmax"]])

        needed_maps = TestRec.rec._needed_amount_maps(
                    cv["filmstart"], cv["filmstop"],
                    cv["filmstep"], cv["profile_count"], mi)

        self.assertEqual(needed_maps, rv["needed_maps"],
                         msg="Error in calculation of needed maps")

    def test_init_arrays(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        maps, mapsi, mapsw = TestRec.rec._init_arrays(
                                cv["profile_count"], cv["reb_profile_length"],
                                rv["needed_maps"], rv["fmlistlength"])

        # All of these should be equal to zero
        self.assertFalse(maps.any())
        self.assertFalse((mapsi + 1).any())
        self.assertFalse(mapsw.any())

        # Checking the shape of the arrays
        self.assertEqual(maps.shape, (100, 205, 205),
                         msg="Error in shape of maps array")
        self.assertEqual(mapsi.shape, (2539200, 16),
                         msg="Error in shape of mapsi array")
        self.assertEqual(mapsw.shape, (2539200, 16),
                         msg="Error in shape of mapsweight array")

    def test_first_map(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays

        maps = np.zeros((cv["profile_count"],
                         cv["reb_profile_length"],
                         cv["reb_profile_length"]), int)
        mapsi = np.full((rv["needed_maps"], rv["fmlistlength"]), -1, int)
        mapsweight = np.zeros((rv["needed_maps"], rv["fmlistlength"]), int)

        (maps[0, :, :], mapsi,
         mapsweight, actmaps) = Reconstruct._first_map(
                                    ca["imin"][0], ca["imax"][0],
                                    ca["jmin"], ca["jmax"],
                                    cv["snpt"]**2,
                                    maps[0], mapsi, mapsweight)
        nptest.assert_equal(maps[0], rv["maps"][0],
                            err_msg="Error in calculation of first maps")
        nptest.assert_equal(mapsi[0:25392], rv["mapsi"][0:25392],
                            err_msg="Error in calculation of first mapsi")
        nptest.assert_equal(mapsweight[0:25392], rv["mapsw"][0:25392],
                            err_msg="Error in calculation of first mapsweight")
        self.assertEqual(actmaps, 25392,
                         msg="Error in number of maps used")

    def test_init_tracked_point(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays

        xp = np.zeros(int(np.ceil(rv["needed_maps"] * cv["snpt"]**2
                                  / cv["profile_count"])))
        yp = np.zeros(int(np.ceil(rv["needed_maps"] * cv["snpt"]**2
                                  / cv["profile_count"])))

        (xp, yp,
         last_pxlidx) = Reconstruct._init_tracked_point(
                            cv["snpt"], ca["imin"][0],
                            ca["imax"][0], ca["jmin"],
                            ca["jmax"], xp, yp,
                            rv["points"][0], rv["points"][1])

        nptest.assert_equal(xp, rv["init_xp"],
                            err_msg="Error in initiating of tracked points (xp)")
        nptest.assert_almost_equal(yp, rv["init_yp"],
                                   err_msg="Error in initiating of tracked points (yp)")
        self.assertEqual(last_pxlidx, 406272,
                         msg="Error in number of used pixels")

    def test_longtrack_forward(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays
        direction = 1
        turn_now = 0
        xp, yp, turn_now = Reconstruct.longtrack(
                                direction, cv["dturns"],
                                rv["init_yp"][:406272],
                                rv["init_xp"][:406272],
                                cv["debin"], turn_now, cv["xorigin"],
                                cv["h_num"], ca["omegarev0"], cv["dtbin"],
                                ca["phi0"], cv["yat0"], ca["dphase"],
                                ca["deltaE0"], cv["vrf1"], cv["vrf1dot"],
                                cv["vrf2"], cv["vrf2dot"], ca["time_at_turn"],
                                cv["hratio"], cv["phi12"], cv["q"])
        nptest.assert_almost_equal(xp, rv["first_xp"],
                                   err_msg="Error after forward "
                                           "longtrack in xp array")
        nptest.assert_almost_equal(yp, rv["first_yp"],
                                   err_msg="Error after forward "
                                           "longtrack in yp array")
        self.assertEqual(turn_now, 12,
                         msg="Error in number of turns iterated through")

    def test_longtrack_backward(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays
        direction = -1
        turn_now = 12

        xp, yp, turn_now = Reconstruct.longtrack(
                                direction, cv["dturns"],
                                rv["init_yp"][:406272],
                                rv["init_xp"][:406272],
                                cv["debin"], turn_now, cv["xorigin"],
                                cv["h_num"], ca["omegarev0"], cv["dtbin"],
                                ca["phi0"], cv["yat0"], ca["dphase"],
                                ca["deltaE0"], cv["vrf1"], cv["vrf1dot"],
                                cv["vrf2"], cv["vrf2dot"], ca["time_at_turn"],
                                cv["hratio"], cv["phi12"], cv["q"])

        nptest.assert_equal(xp, rv["first_rev_xp"],
                            err_msg="Error after longtrack in xp array")
        nptest.assert_equal(yp, rv["first_rev_yp"],
                            err_msg="Error after longtrack in yp array")
        self.assertEqual(turn_now, 0,
                         msg="Error in number of turns iterated through")

    def test_longtrack_self_forward(self):

        # To be written
        pass

    def test_longtrack_self_backward(self):

        # To be written
        pass

    def test_calc_weightfactors(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays
        actmaps = 25392

        maps = np.zeros((2, 205, 205), int)
        mapsi = np.full((50784, 16), -1, int)
        mapsw = np.zeros((50784, 16), int)

        maps[0] = rv["maps"][0].copy()
        mapsi[:actmaps] = rv["mapsi"][:actmaps].copy()
        mapsw[:actmaps] = rv["mapsw"][:actmaps].copy()

        isOut, actmaps = Reconstruct._calc_weightfactors(
                            ca["imin"][0], ca["imax"][0],
                            ca["jmin"], ca["jmax"],
                            maps[1, :, :], mapsi[:, :],
                            mapsw[:, :], rv["first_xp"],
                            cv["snpt"]**2, cv["reb_profile_length"],
                            rv["fmlistlength"], actmaps)

        nptest.assert_equal(maps[1, :, :], rv["maps"][1, :, :],
                            err_msg="Error in calculation of maps")
        nptest.assert_equal(mapsi[:50784], rv["mapsi"][:50784],
                            err_msg="Error in calculation of mapsi")
        nptest.assert_equal(mapsw[:50784], rv["mapsw"][:50784],
                            err_msg="Error in calculation of mapsweights")
        self.assertEqual(actmaps, 50784,
                         msg="Error in number of maps used")
        self.assertEqual(isOut, 0,
                         "Error in number of lost data")

    def test_total_weightfactor(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays
        reversedweights = Reconstruct._total_weightfactor(
                            cv["profile_count"],
                            cv["reb_profile_length"],
                            rv["fmlistlength"], cv["snpt"]**2,
                            ca["imin"][0],
                            ca["imax"][0],
                            ca["jmin"],
                            ca["jmax"],
                            rv["mapsw"], rv["mapsi"], rv["maps"])
        nptest.assert_almost_equal(reversedweights, rv["rweights"],
                                   err_msg="Error in calculation "
                                           "of reversed weights")


if __name__ == '__main__':
    unittest.main()
