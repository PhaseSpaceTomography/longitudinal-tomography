import unittest
import numpy.testing as nptest
import numpy as np
import ctypes
import os
from numpy.ctypeslib import ndpointer
from tomo.reconstruct_c import ReconstructCpp
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
        cls.rec = ReconstructCpp.__new__(ReconstructCpp)

        # tomolib = ctypes.CDLL(
        #     os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])
        #     + '/cpp_files/tomolib.so')

        tomolib = ctypes.CDLL('./../tomo/cpp_files/tomolib.so')

        tomolib.weight_factor_array.argtypes = [ndpointer(ctypes.c_double),
                                                ndpointer(ctypes.c_int),
                                                ndpointer(ctypes.c_int),
                                                ndpointer(ctypes.c_int),
                                                ndpointer(ctypes.c_int),
                                                ndpointer(ctypes.c_int),
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.c_int]

        tomolib.first_map.argtypes = [ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ndpointer(ctypes.c_int),
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int]

        tomolib.longtrack.argtypes = [ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ndpointer(ctypes.c_double),
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_double,
                                      ctypes.c_int,
                                      ctypes.c_double]

        cls.find_mapweight = tomolib.weight_factor_array
        cls.first_map = tomolib.first_map
        cls.longtrack_cpp = tomolib.longtrack

    @classmethod
    def tearDownClass(cls):
        # Getting correct values and arrays for C500MidPhaseNoise input
        del cls.c500
        del cls.rec_vals
        # Making MapInfo object for calling functions
        del cls.rec

    def test_number_of_maps(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays
        mi = MapInfo.__new__(MapInfo)
        mi.imin = ca["imin"]
        mi.imax = ca["imax"]
        mi.jmin = np.array([ca["jmin"]])
        mi.jmax = np.array([ca["jmax"]])

        needed_maps = TestRec.rec._submaps_needed(
                        cv["filmstart"], cv["filmstop"],
                        cv["filmstep"], cv["profile_count"],
                        ca["imin"], ca["imax"],
                        np.array([ca["jmin"]]),
                        np.array([ca["jmax"]]))

        self.assertEqual(needed_maps, rv["needed_maps"],
                         msg="Error in calculation of needed maps")

    def test_populate_bins(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        points = TestRec.rec._populate_bins(cv["snpt"])
        nptest.assert_equal(points, rv["points"],
                            err_msg="Initial points calculated incorrectly.")

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
        self.assertEqual(maps.shape, (100, 42025),
                         msg="Error in shape of maps array")
        self.assertEqual(mapsi.shape, (40627200, ),
                         msg="Error in shape of mapsi array")
        self.assertEqual(mapsw.shape, (40627200, ),
                         msg="Error in shape of mapsweight array")

    def test_init_tracked_point(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays

        xp = np.zeros(int(np.ceil(rv["needed_maps"] * cv["snpt"]**2
                                  / cv["profile_count"])))
        yp = np.zeros(int(np.ceil(rv["needed_maps"] * cv["snpt"]**2
                                  / cv["profile_count"])))

        (xp, yp,
         last_pxlidx) = ReconstructCpp._init_tracked_point(
                            cv["snpt"], ca["imin"][0],
                            ca["imax"][0], ca["jmin"],
                            ca["jmax"], xp, yp,
                            rv["points"][0], rv["points"][1])

        nptest.assert_equal(xp, rv["init_xp"],
                            err_msg="Error in initiating"
                                    " of tracked points (xp)")
        nptest.assert_almost_equal(yp, rv["init_yp"],
                                   err_msg="Error in initiating "
                                           "of tracked points (yp)")
        self.assertEqual(last_pxlidx, 406272,
                         msg="Error in number of used pixels")

    def test_longtrack_forward(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays
        direction = 1
        turn_now = 0

        xp = np.copy(rv["init_xp"][:406272])
        yp = np.copy(rv["init_yp"][:406272])

        turn_now = self.longtrack_cpp(
                                xp, yp, ca["omegarev0"], ca["phi0"], ca["dphase"],
                                ca["time_at_turn"], ca["deltaE0"],
                                np.int32(406272), cv["xorigin"], cv["dtbin"],
                                cv["debin"], cv["yat0"], cv["phi12"], direction,
                                cv["dturns"], turn_now, cv["q"], cv["vrf1"],
                                cv["vrf1dot"], cv["vrf2"], cv["vrf2dot"],
                                np.int32(cv["h_num"]),
                                np.float64(cv["hratio"]))

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

        xp = np.copy(rv["init_xp"][:406272])
        yp = np.copy(rv["init_yp"][:406272])

        turn_now = self.longtrack_cpp(
                                xp, yp, ca["omegarev0"], ca["phi0"], ca["dphase"],
                                ca["time_at_turn"], ca["deltaE0"],
                                np.int32(406272), cv["xorigin"], cv["dtbin"],
                                cv["debin"], cv["yat0"], cv["phi12"], direction,
                                cv["dturns"], turn_now, cv["q"], cv["vrf1"],
                                cv["vrf1dot"], cv["vrf2"], cv["vrf2dot"],
                                np.int32(cv["h_num"]),
                                np.float64(cv["hratio"]))

        nptest.assert_almost_equal(xp, rv["first_rev_xp"],
                                   err_msg="Error after longtrack in xp array")
        nptest.assert_almost_equal(yp, rv["first_rev_yp"],
                                   err_msg="Error after longtrack in yp array")
        self.assertEqual(turn_now, 0,
                         msg="Error in number of turns iterated through")

    def test_total_weightfactor(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays
        reversedweights = ReconstructCpp._total_weightfactor(
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

    def test_calc_weightfactors(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays

        submap = 25392

        maps = np.zeros((2, 205, 205), dtype=np.int32)
        mapsi = np.full((50784, 16), -1, dtype=np.int32)
        mapsw = np.zeros((50784, 16), dtype=np.int32)

        maps[0] = rv["maps"][0].copy()
        mapsi[:submap] = rv["mapsi"][:submap].copy()
        mapsw[:submap] = rv["mapsw"][:submap].copy()

        maps = np.reshape(maps, (2, 42025))
        mapsi = np.reshape(mapsi, (812544, ))
        mapsw = np.reshape(mapsw, (812544, ))

        is_out = self.find_mapweight(
                            rv["first_xp"],
                            np.array(ca["jmin"], dtype=np.int32),
                            np.array(ca["jmax"], dtype=np.int32),
                            maps[1], mapsi, mapsw,
                            np.int32(ca["imin"][0]),
                            np.int32(ca["imax"][0]),
                            np.int32(cv["snpt"]**2),
                            np.int32(cv["reb_profile_length"]),
                            np.int32(cv["snpt"]**2),
                            submap)

        maps = np.reshape(maps, (2, 205, 205))
        mapsi = np.reshape(mapsi, (50784, 16))
        mapsw = np.reshape(mapsw, (50784, 16))

        nptest.assert_equal(maps[1, :, :], rv["maps"][1, :, :],
                            err_msg="Error in calculation of maps")
        nptest.assert_equal(mapsi[:50784], rv["mapsi"][:50784],
                            err_msg="Error in calculation of mapsi")
        nptest.assert_equal(mapsw[:50784], rv["mapsw"][:50784],
                            err_msg="Error in calculation of mapsweights")
        self.assertEqual(is_out, 0,
                         "Error in number of lost data")

    def test_calc_first_map(self):
        rv = TestRec.rec_vals
        cv = TestRec.c500.values
        ca = TestRec.c500.arrays

        maps = np.zeros((cv["profile_count"],
                         cv["reb_profile_length"] * cv["reb_profile_length"]),
                        dtype=np.int32)

        mapsi = np.full((rv["needed_maps"] * rv["fmlistlength"]), -1, dtype=np.int32)

        mapsw = np.zeros((rv["needed_maps"] * rv["fmlistlength"]), dtype=np.int32)

        jmin = np.array(ca["jmin"], dtype=np.int32)
        jmax = np.array(ca["jmax"], dtype=np.int32)

        nr_of_submaps = self.first_map(
                            jmin, jmax, maps[0], mapsi, mapsw,
                            ca["imin"][0], ca["imax"][0],
                            cv["snpt"]**2, cv["reb_profile_length"])

        maps = maps.reshape((cv["profile_count"],
                             cv["reb_profile_length"],
                             cv["reb_profile_length"]))
        mapsi = mapsi.reshape((rv["needed_maps"],
                               rv["fmlistlength"]))
        mapsw = mapsw.reshape((rv["needed_maps"],
                               rv["fmlistlength"]))

        nptest.assert_equal(maps[0], rv["maps"][0],
                            err_msg="Error in calculation of first maps")
        nptest.assert_equal(mapsi[0:25392], rv["mapsi"][0:25392],
                            err_msg="Error in calculation of first mapsi")
        nptest.assert_equal(mapsw[0:25392], rv["mapsw"][0:25392],
                            err_msg="Error in calculation of first mapsweight")
        self.assertEqual(nr_of_submaps, 25392,
                         msg="Error in number of maps used")


if __name__ == '__main__':
    unittest.main()
