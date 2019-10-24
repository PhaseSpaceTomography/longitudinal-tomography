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
        
        input_file = cls.c500.path + 'C500MidPhaseNoise.dat'
        with open(input_file, 'r') as f:
            read = f.readlines()
        raw_data = np.array(read[98:], dtype=float)
        par = Parameters()

        # To be filled manually using values from cv and ca?
        par.parse_from_txt(read[:98])
        par.fill()

        ts = TimeSpace()
        ts.create(par, raw_data)

        # Creating map info object (uninitialized)
        cls.mi = MapInfo.__new__(MapInfo)
        cls.mi.par = ts.par

    def test_energy_binning_C500(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays
        turn = (cv['beam_ref_frame'] - 1) * cv['dturns']

        dEbin = self.mi.find_dEbin(ca['phases'], turn)

        self.assertAlmostEqual(dEbin, cv['debin'],
                               msg='dEbin was not correctly calculated')

    def test_calc_phases_C500(self):
        ca = TestMap.c500.arrays
        turn = 0

        phases = self.mi.calculate_phases(turn)

        nptest.assert_almost_equal(phases, ca['phases'],
                                   err_msg='Difference in '
                                           'calculation of phases')

    def test_trajectory_height_arr_C500(self):
        ca = TestMap.c500.arrays
        energy = 0.0
        turn = 0
        test_ans = 287470.8666145134

        ans = self.mi.trajectoryheight(ca['phases'], ca['phases'][0],
                                       energy, turn)

        self.assertAlmostEqual(ans[0], 0.0,
                               msg='trajectory height (array): '
                                   'first element not correct')
        self.assertAlmostEqual(ans[1], test_ans,
                               msg='trajectory height (array): '
                                   'second element not correct')
        self.assertEqual(len(ca['phases']), len(ans),
                         msg='trajectory height (array): '
                             'length of array not correct')

    def test_trajectory_height_val_C500(self):
        ca = TestMap.c500.arrays
        energy = 0.0
        turn = 0
        expected_ans = 287470.8666145134

        ans = self.mi.trajectoryheight(ca['phases'][1], ca['phases'][0],
                                       energy, turn)

        self.assertAlmostEqual(ans, expected_ans,
                               msg='trajectory height not '
                                   'calculated correctly')

    def test_track_all_limits(self):
        (jmin, jmax,
         allbin_min,
         allbin_max) = self.mi._limits_track_all_pxl()

        self.assertEqual(allbin_min, [0], msg='Error in allbin_min '
                                              '(track all pxl)')
        self.assertEqual(allbin_max, [205], msg='Error in allbin_max '
                                                '(track all pxl)')
        nptest.assert_equal(jmin, np.full(205, 1, dtype=int),
                            err_msg='Error in calculation of jmin '
                                    '(track all pxl)')
        nptest.assert_equal(jmax, np.full(205, 205, dtype=int),
                            err_msg='Error in calculation of jmax '
                                    '(track all pxl)')

    def test_track_active_pxl(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays

        (jmin,
         jmax,
         allbin_min,
         allbin_max) = self.mi._limits_track_active_pxl(cv['debin'])

        nptest.assert_equal(jmin, ca['jmin'].flatten(),
                            err_msg='Error in calculation of jmin')
        nptest.assert_equal(jmax, ca['jmax'].flatten(),
                            err_msg='Error in calculation of jmax')
        self.assertEqual(allbin_min, cv['allbin_min'],
                         msg='Error in calculation of allbin_min')
        self.assertEqual(allbin_max, cv['allbin_max'],
                         msg='Error in calculation of allbin_max')

    def test_find_jmax(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays
        turn = 0

        jmax = self.mi._find_jmax(ca['phases'], turn, cv['debin'])

        nptest.assert_equal(jmax, ca['jmax'],
                            err_msg='find jmax function failed')

    def test_find_jmin(self):
        ca = TestMap.c500.arrays

        jmin = self.mi._find_jmin(ca['jmax'])

        nptest.assert_equal(jmin, ca['jmin'],
                            err_msg='find jmin function failed')

    def test_find_allbin_min(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays

        allbin_min = self.mi._find_allbin_min(ca['jmin'], ca['jmax'])

        nptest.assert_equal(allbin_min, cv['allbin_min'],
                            err_msg='Find allbin_min function failed')

    def test_find_allbin_max(self):
        cv = TestMap.c500.values
        ca = TestMap.c500.arrays

        allbin_max = self.mi._find_allbin_max(ca['jmin'], ca['jmax'])

        nptest.assert_equal(allbin_max, cv['allbin_max'],
                            err_msg='Find allbin_max function failed')

    def test_bad_ilimits(self):
        self.mi.jmin = np.array([self.c500.arrays['jmin']])
        self.mi.jmax = np.array([self.c500.arrays['jmax']])

        # Too low imin
        self.mi.imin = np.array([-1])
        self.mi.imax = self.c500.arrays['imax']
        with self.assertRaises(Exception):
            self.mi._assert_correct_arrays()

        # Too low imax
        self.mi.imin = self.c500.arrays['imin']
        self.mi.imax = np.array([1])
        with self.assertRaises(Exception):
            self.mi._assert_correct_arrays()

        self.mi.imin = self.c500.arrays['imin']
        self.mi.imax = np.array([self.c500.arrays['jmax'].size + 1])
        with self.assertRaises(Exception):
            self.mi._assert_correct_arrays()

    def test_bad_jlimits(self):
        self.mi.jmin = np.array([self.c500.arrays['jmin']])
        self.mi.jmax = np.array([self.c500.arrays['jmax']])
        self.mi.imin = self.c500.arrays['imin']
        self.mi.imax = self.c500.arrays['imax']

        # Too low jmin
        self.mi.jmin[0, 100] = -3
        with self.assertRaises(Exception):
            self.mi._assert_correct_arrays()
        self.mi.jmin = np.array([self.c500.arrays['jmin']])

        # Too low jmax
        self.mi.jmax[0, 5] = 50
        with self.assertRaises(Exception):
            self.mi._assert_correct_arrays()
        self.mi.jmax = np.array([self.c500.arrays['jmax']])

        # Too high jmax
        self.mi.jmax[0, 177] = self.mi.par.profile_length + 1
        with self.assertRaises(Exception):
            self.mi._assert_correct_arrays()


if __name__ == '__main__':
    unittest.main()

