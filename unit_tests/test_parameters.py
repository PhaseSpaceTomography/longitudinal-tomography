import unittest
import numpy as np
import numpy.testing as nptest
from tomo.parameters import Parameters
from unit_tests.C500values import C500


class TestParameters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c500 = C500()

    @classmethod
    def tearDownClass(cls):
        del cls.c500

    def setUp(self):
        self.par = Parameters()
        self.par.get_parameters_txt(self.c500.path + "C500MidPhaseNoise.dat")

    def test_variable_types(self):
        self.assertIsInstance(self.par.xat0, float,
                              msg="xAt0 is not a float")
        self.assertIsInstance(self.par.yat0, float,
                              msg="yAt0 is not a float")
        self.assertIsInstance(self.par.rebin, int,
                              msg="rebin is not an int")
        self.assertIsInstance(self.par.rawdata_file, str,
                              msg="rawDataFile is not a string")
        self.assertIsInstance(self.par.output_dir, str,
                              msg="outputDir is not a string")
        self.assertIsInstance(self.par.framecount, int,
                              msg="frameCount is not an int")
        self.assertIsInstance(self.par.framelength, int,
                              msg="frameLength is not an int")
        self.assertIsInstance(self.par.dtbin, float,
                              msg="dtBin is not a float")
        self.assertIsInstance(self.par.demax, float,
                              msg="dEmax is not a float")
        self.assertIsInstance(self.par.dturns, int,
                              msg="dTurns is not an int")
        self.assertIsInstance(self.par.preskip_length, int,
                              msg="preSkipLength is not an int")
        self.assertIsInstance(self.par.postskip_length, int,
                              msg="postSkipLength is not an int")
        self.assertIsInstance(self.par.imin_skip, int,
                              msg="iMinSkip is not an int")
        self.assertIsInstance(self.par.imax_skip, int,
                              msg="iMaxSkip is not an int")
        self.assertIsInstance(self.par.framecount, int,
                              msg="frameCount is not an int")
        self.assertIsInstance(self.par.frame_skipcount, int,
                              msg="frameSkipCount is not an int")
        self.assertIsInstance(self.par.framelength, int,
                              msg="frameLength is not an int")
        self.assertIsInstance(self.par.snpt, int,
                              msg="sqrtNumPt is not an int")
        self.assertIsInstance(self.par.num_iter, int,
                              msg="nIterations is not an int")
        self.assertIsInstance(self.par.machine_ref_frame, int,
                              msg="machineRefFrame is not an int")
        self.assertIsInstance(self.par.beam_ref_frame, int,
                              msg="beamRefFrame is not an int")
        self.assertIsInstance(self.par.filmstep, int,
                              msg="filmStep is not an int")
        self.assertIsInstance(self.par.filmstart, int,
                              msg="filmStart is not an int")
        self.assertIsInstance(self.par.filmstop, int,
                              msg="filmStop is not an int")
        self.assertIsInstance(self.par.full_pp_flag, bool,
                              msg="fullPPFlag is not a bool")
        self.assertIsInstance(self.par.vrf1, float,
                              msg="VRF1 is not a float")
        self.assertIsInstance(self.par.vrf2, float,
                              msg="VRF2 is not a float")
        self.assertIsInstance(self.par.vrf1dot, float,
                              msg="VRF1dot is not a float")
        self.assertIsInstance(self.par.vrf2dot, float,
                              msg="VRF2dot is not a float")
        self.assertIsInstance(self.par.mean_orbit_rad, float,
                              msg="meanOrbitR is not a float")
        self.assertIsInstance(self.par.bending_rad, float,
                              msg="bendingR is not a float")
        self.assertIsInstance(self.par.b0, float,
                              msg="B0 is not a float")
        self.assertIsInstance(self.par.bdot, float,
                              msg="Bdot is not a float")
        self.assertIsInstance(self.par.phi12, float,
                              msg="phi12 is not a float")
        self.assertIsInstance(self.par.h_ratio, float,
                              msg="hRatio is not a float")
        self.assertIsInstance(self.par.h_num, float,
                              msg="harmonicNum is not a float")
        self.assertIsInstance(self.par.trans_gamma, float,
                              msg="gammaTrans is not a float")
        self.assertIsInstance(self.par.e_rest, float,
                              msg="Erest is not a float")
        self.assertIsInstance(self.par.q, float,
                              msg="q is not a float")
        self.assertIsInstance(self.par.self_field_flag, bool,
                              msg="selfFieldFlag is not a bool")
        self.assertIsInstance(self.par.g_coupling, float,
                              msg="gCoupling is not a float")
        self.assertIsInstance(self.par.zwall_over_n, float,
                              msg="zWallOverN is not a float")
        self.assertIsInstance(self.par.pickup_sensitivity, float,
                              msg="pickUpSensitivity is not a float")

    def test_initialization_of_arrays_C500_inn(self):

        allturns = ((self.par.framecount - self.par.frame_skipcount - 1)
                    * self.par.dturns)

        self.assertEqual(len(self.par.time_at_turn), allturns + 1,
                         msg="Error in length of array: time_at_turn ")

        # Calculating initial index
        i0 = (self.par.machine_ref_frame - 1) * self.par.dturns

        # Testing initial values of arrays
        self.assertEqual(self.par.time_at_turn[i0], 0,
                         msg="Error in initial value of time_at_turn array")
        self.assertAlmostEqual(self.par.e0[i0], 1335012859.7126765,
                               msg="Error in initial value of e0 array")
        self.assertAlmostEqual(self.par.beta0[i0], 0.7113687870661543,
                               msg="Error in initial value of beta0 array")
        self.assertAlmostEqual(self.par.phi0[i0], 0.3116495273168101,
                               msg="Error in initial value of phi0 array")

    def test_remaining_array_values_C500_inn(self):
        ca = self.c500.arrays

        nptest.assert_almost_equal(self.par.beta0, ca["beta0"],
                                   err_msg="Error in calculation of"
                                           " beta0 array")
        nptest.assert_almost_equal(self.par.deltaE0, ca["deltaE0"],
                                   err_msg="Error in calculation of"
                                           " deltaE0 array")
        nptest.assert_almost_equal(self.par.e0, ca["e0"],
                                   err_msg="Error in calculation of"
                                           " e0 array")
        nptest.assert_almost_equal(self.par.time_at_turn, ca["time_at_turn"],
                                   err_msg="Error in calculation of"
                                           " time_at_turn array")
        nptest.assert_almost_equal(self.par.phi0, ca["phi0"],
                                   err_msg="Error in calculation of"
                                           " phi0 array")
        nptest.assert_almost_equal(self.par.eta0, ca["eta0"],
                                   err_msg="Error in calculation of"
                                           " eta0 array")
        nptest.assert_almost_equal(self.par.c1, ca["c1"],
                                   err_msg="Error in calculation of"
                                           " c1 array")
        nptest.assert_almost_equal(self.par.omega_rev0, ca["omegarev0"],
                                   err_msg="Error in calculation of"
                                           " omega_rev0 array")

    def test_remaining_parameters_C500_inn(self):
        ca = self.c500.arrays

        self.assertAlmostEqual(self.par.dtbin, 1.9999999999999997e-09,
                               msg="Error in calculation of dtbin")
        self.assertAlmostEqual(self.par.xat0, 88.00000000000001,
                               msg="Error in calculation of xat0")
        self.assertEqual(self.par.profile_count, 100,
                         msg="Error in calculation of profile_count")
        self.assertEqual(self.par.profile_mini, 0,
                         msg="Error in calculation of profile_mini")
        self.assertEqual(self.par.profile_maxi, 205,
                         msg="Error in calculation of profile_maxi")
        self.assertEqual(self.par.all_data, 100000,
                         msg="Error in calculation of all_data")
        nptest.assert_almost_equal(self.par.sfc, ca["sfc"],
                                   err_msg="Error in calculation of sfc array")

    # TODO: Later, when pipelining is implemented, make test
    #  to verify that the asserion funtions are actually implemented
    # Testing bad input
    def test_bad_framecount(self):
        # Testing frame count
        temp = self.par.framecount
        self.par.framecount = 0
        with self.assertRaises(Exception):
            self.par._assert_parameters()
        self.par.framecount = temp

    def test_bad_frame_skipcount(self):
        self.par.frame_skipcount = -1
        with self.assertRaises(Exception):
            self.par._assert_input()
        self.par.frame_skipcount = self.par.framecount + 1
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_frame_length(self):
        self.par.framelength = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_preskip_length(self):
        self.par.preskip_length = -1
        with self.assertRaises(Exception):
            self.par._assert_input()
        self.par.preskip_length = self.par.framelength + 1
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_postskip_length(self):
        self.par.postskip_length = -1
        with self.assertRaises(Exception):
            self.par._assert_input()
        self.par.postskip_length = self.par.framelength + 1
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_dtbin(self):
        self.par.dtbin = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_dturns(self):
        self.par.dturns = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_imin_skip(self):
        self.par.imin_skip = -1
        with self.assertRaises(Exception):
            self.par._assert_input()
        self.par.imin_skip = self.par.framelength + 1
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_imax_skip(self):
        self.par.imax_skip = -1
        with self.assertRaises(Exception):
            self.par._assert_input()
        self.par.imax_skip = self.par.framelength + 1
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_rebin(self):
        self.par.rebin = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_filmstart(self):
        self.par.filmstart = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_filmstop(self):
        self.par.filmstop = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_filmstep(self):
        self.par.filmstep = 0
        with self.assertRaises(Exception):
            self.par._assert_input()
        self.par.filmstep = 2
        with self.assertRaises(Exception):
            self.par._assert_input()
        self.par.filmstep = -2

    def test_bad_num_iter(self):
        self.par.num_iter = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_snpt(self):
        self.par.snpt = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_machine_ref_frame(self):
        self.par.machine_ref_frame = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_beam_ref_frame(self):
        self.par.beam_ref_frame = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_h_num(self):
        self.par.h_num = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_h_ratio(self):
        self.par.h_ratio = 0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_b0(self):
        self.par.b0 = 0.0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_mean_orbit_rad(self):
        self.par.mean_orbit_rad = 0.0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_bending_rad(self):
        self.par.bending_rad = 0.0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_e_rest(self):
        self.par.e_rest = 0.0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_zwall_over_n(self):
        self.par.zwall_over_n = -0.01
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_pickup_sensitivity(self):
        self.par.pickup_sensitivity = -1.0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_g_coupling(self):
        self.par.g_coupling = -10.0
        with self.assertRaises(Exception):
            self.par._assert_input()

    def test_bad_profile_length(self):
        self.par.profile_length = -1
        with self.assertRaises(Exception):
            self.par._assert_parameters()

    def test_bad_array_lengths(self):
        # Testing time at turn
        temp = np.copy(self.par.time_at_turn)
        self.par.time_at_turn = np.zeros(0)
        with self.assertRaises(Exception):
            self.par._assert_parameters()
        self.par.time_at_turn = temp

        # testing omega_rev0
        temp = np.copy(self.par.omega_rev0)
        self.par.omega_rev0 = np.zeros(10000)
        with self.assertRaises(Exception):
            self.par._assert_parameters()
        self.par.omega_rev0 = temp

        # testing phi0
        temp = np.copy(self.par.phi0)
        self.par.phi0 = np.zeros((5, 1024))
        with self.assertRaises(Exception):
            self.par._assert_parameters()
        self.par.phi0 = temp

        # testing c1
        temp = np.copy(self.par.c1)
        self.par.c1 = np.zeros((1189, 1))
        with self.assertRaises(Exception):
            self.par._assert_parameters()
        self.par.c1 = temp

        # testing deltaE0
        temp = np.copy(self.par.deltaE0)
        self.par.deltaE0 = np.zeros((1189, 0))
        with self.assertRaises(Exception):
            self.par._assert_parameters()
        self.par.deltaE0 = temp

        # testing beta0
        temp = np.copy(self.par.beta0)
        self.par.beta0 = np.zeros((0, 1189))
        with self.assertRaises(Exception):
            self.par._assert_parameters()
        self.par.beta0 = temp

        # testing eta0
        temp = np.copy(self.par.eta0)
        self.par.eta0 = np.zeros((0, 1189))
        with self.assertRaises(Exception):
            self.par._assert_parameters()
        self.par.eta0 = temp

        # testing e0
        temp = np.copy(self.par.e0)
        self.par.e0 = np.zeros((0, 1189))
        with self.assertRaises(Exception):
            self.par._assert_parameters()
        self.par.e0 = temp


if __name__ == '__main__':
    unittest.main()
