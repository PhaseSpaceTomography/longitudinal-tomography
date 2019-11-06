import unittest
import numpy as np
import numpy.testing as nptest
from tomo.parameters import Parameters
from tomo.time_space import TimeSpace
from unit_tests.C500values import C500

# All tests marked with C500 takes hold in values
#   from the C500MidPhaseNoise input

class TestTimeSpace(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c500 = C500()

        input_file = cls.c500.path + 'C500MidPhaseNoise.dat'
        with open(input_file, 'r') as f:
            read = f.readlines()
        
        cls.raw_param = read[:98]
        cls.raw_data = np.array(read[98:], dtype=float)

    def setUp(self):
        par = Parameters()
        par.parse_from_txt(self.raw_param)
        par.fill()
        self.timespace = TimeSpace(par)

    # Testing raw-data inn
    def test_subtract_baseline_C500(self):
        ca = TestTimeSpace.c500.arrays
        res = self.timespace.subtract_baseline(self.raw_data)
        correct = ca['data_baseline_subtr']
        nptest.assert_almost_equal(
                res, correct,err_msg='Error in calculation of baseline')

    # Testing conversion from raw-data to finished profile
    def test_rawdata_to_profiles_C500(self):
        ca = TestTimeSpace.c500.arrays
        data_subtr_baseln = ca['data_baseline_subtr']
        
        res = self.timespace.rawdata_to_profiles(data_subtr_baseln)
        correct = ca['raw_profiles']

        nptest.assert_almost_equal(
            res.flatten(), correct.flatten(),
            err_msg='Error in conversion from raw data to profiles')

    # Test rebinning with even rebinning factor
    # In the original C500MidPhaseNoise, the rebin factor is even.
    def test_rebinning_C500_even(self):
        ca = TestTimeSpace.c500.arrays
        cv = TestTimeSpace.c500.values

        res, res_nbins = self.timespace.rebin(ca['raw_profiles'])
        
        nptest.assert_almost_equal(
            res, ca['rebinned_profiles_even'],
            err_msg='Error in re-binning of profiles (even rebin factor)')

        self.assertEqual(
            res_nbins, cv['reb_profile_length'],
            msg='Error in length of re-binned profiles (even rebin factor)')

    def test_rebinning_C500_odd(self):
        ca = TestTimeSpace.c500.arrays
        cv = TestTimeSpace.c500.values

        self.timespace.par.rebin = 5
        res, res_nbins = self.timespace.rebin(ca['raw_profiles'])
        
        nptest.assert_almost_equal(
            res, ca['rebinned_profiles_odd'],
            err_msg='Error in re-binning of profiles (odd rebin factor)')

        expected_nbins = 164
        self.assertEqual(
            res_nbins, expected_nbins,
            msg='Error in length of re-binned profiles (odd rebin factor)')

    def test_finished_profile_C500(self):
        ca = TestTimeSpace.c500.arrays
        correct = ca['profiles']
        
        res , _ = self.timespace.create_profiles(self.raw_data)

        print(res.shape)
        print(correct.shape)
        
        np.testing.assert_almost_equal(
            res, correct, err_msg='Error in creation of profiles')

    # Testing calculation of general TimeSpace variables
    def test_profile_charge_C500(self):
        ca = TestTimeSpace.c500.arrays

        ref_prof = 0
        profiles = ca['rebinned_profiles_even'].clip(0.0)
        correct = 1597936374926.2097

        res = self.timespace.calc_profilecharge(profiles[ref_prof])

        self.assertAlmostEqual(
            res, correct, msg='Error in calculation of profile charge')

    def test_calc_xorigin_C500(self):
        cv = TestTimeSpace.c500.values
        ca = TestTimeSpace.c500.arrays
        res = self.timespace.calc_xorigin()
        correct = cv['xorigin']
        self.assertAlmostEqual(
            res, correct, msg='Error in calculation of xorigin')

    def test_fit_xat0_C500(self):
        cv = TestTimeSpace.c500.values
        ca = TestTimeSpace.c500.arrays
    
        # Function needs rebinned profiles and the profile charge 
        self.timespace.profiles = ca['profiles']
        self.timespace.profile_charge = 1597936374926.2097
        
        # When the function is called,
        # sthe given xat0 parameter will be negative 
        self.xat0 = -1
    
        fit_xat0, ltfoot, utfoot = self.timespace.fit_xat0()
    
        self.assertEqual(utfoot, 186.58843537414958,
            msg='Error in calculation of the upper tangent foot')
        self.assertEqual(ltfoot, 15.464929214929239,
                         msg='Error in calculation of the lower tangent foot')
        self.assertAlmostEqual(fit_xat0, 91.9098144506852,
                               msg='Error in calculation of fitted x at zero')


    def test_calc_tangentbin_C500(self):
        ca = TestTimeSpace.c500.arrays
    
        # Function needs rebinned profiles and the profile charge 
        self.timespace.profile_charge = 1597936374926.2097
        ref_prof = ca['profiles'][0]

        upper, lower = self.timespace._calc_tangentbins(ref_prof)

        self.assertEqual(
            lower, 20, msg='Error in calculation of lower tangent bin')
        self.assertEqual(
            upper, 176, msg='Error in calculation of upper tangent bin')
        self.assertIsInstance(
            lower, int, msg='Lower tangent bin is not integer')
        self.assertIsInstance(
            upper, int, msg='Upper tangent bin is not integer')

    def test_calc_tangentfeet_C500(self):
        ca = TestTimeSpace.c500.arrays

        # Function needs rebinned profiles and the profile charge 
        self.timespace.profile_charge = 1597936374926.2097
        ref_prof = ca['profiles'][0]

        upper, lower = self.timespace.calc_tangentfeet(ref_prof)

        self.assertAlmostEqual(
            lower, 15.464929214929239,
            msg='Error in calculation of lower foot tangent')
        self.assertAlmostEqual(
            upper, 186.58843537414958,
            msg='Error in calculation of upper foot tangent')

    def test_find_wraplength_C500(self):
        cv = TestTimeSpace.c500.values

        # Calculation depends on the profile length.
        # Using the rebinned profile length in order to get the 
        #  same value as in C500MidPhaseNoise
        self.timespace.par.profile_length = cv['reb_profile_length']
        
        phiwrp, wrp_len = self.timespace.find_wrap_length()
        
        self.assertAlmostEqual(phiwrp, cv['phiwrap'],
                               msg='Error in calculation of phiwrap')
        self.assertEqual(wrp_len, cv['wraplength'],
                         msg='Error in calculation of wraplength')

    def test_find_wraplength_C500_bdot_eq0(self):
        cv = TestTimeSpace.c500.values
        
        self.timespace.par.profile_length = cv['reb_profile_length']
        self.timespace.par.bdot = 0

        phiwrp, wrp_len = self.timespace.find_wrap_length()

        self.assertAlmostEqual(phiwrp, 6.283185307179586,
                               msg='Error in calculation of phiwrap '
                                   '(bdot = 0)')
        self.assertEqual(wrp_len, 369,
                         msg='Error in calculation of wraplength '
                             '(bdot = 0)')

    def test_calc_yat0_C500(self):
        cv = TestTimeSpace.c500.values
        self.timespace.par.profile_length = cv['reb_profile_length']
        res = self.timespace.find_yat0()
        correct = 102.5
        self.assertEqual(res, correct, msg='Error in calculation of yat0')

    # Testing too large rebin factor
    def test_bad_rebin(self):
        cv = TestTimeSpace.c500.values
        ca = TestTimeSpace.c500.arrays
        self.timespace.par.rebin = cv['init_profile_length'] + 1
        
        with self.assertRaises(Exception):
            _, _ = self.timespace.rebin(ca['raw_profiles'])


    def test_bad_percentage_input_subtr_baseline(self):
        ca = TestTimeSpace.c500.arrays
        correct = ca['data_baseline_subtr']

        with self.assertRaises(Exception):
            res = self.timespace.subtract_baseline(self.raw_data,
                                                   percentage=-0.2)

        with self.assertRaises(Exception):
            res = self.timespace.subtract_baseline(self.raw_data,
                                                   percentage=1.1)


    # Make into test for the full create funtion
    # -------------------------------------------------------------------
    # I have to find the correct numbers for this test, or delete it
    #  and base these tsts on noiseStructure2.
    # Remember to add a test for the whole setup. 
    # def test_fit_xat0_C500(self):
    #     cv = TestTimeSpace.c500.values
    #     ca = TestTimeSpace.c500.arrays
    #
    #     self.timespace.profiles = ca['profiles']
    #     self.timespace.profile_charge = 1597936374926.2097
    #
    #     fit_xat0, ltfoot, utfoot = self.timespace.fit_xat0()
    #
    #     self.assertEqual(utfoot, cv['tanfoot_up'],
    #         msg='Error in calculation of tangentfoot_up')
    #     self.assertEqual(ts.par.tangentfoot_low, cv['tanfoot_low'],
    #                      msg='Error in calculation of tangentfoot_low')
    #     self.assertAlmostEqual(fit_xat0, cv['fit_xat0'],
    #                            msg='Error in calculation of fit_xat0')
    # -------------------------------------------------------------------

    # Make into test for full process from raw input to finished profiles 
    # -------------------------------------------------------------------
    # def test_all_negative_raw_data(self):
    #     negative_array = -np.ones(100)
    #     with self.assertRaises(Exception):
    #         TimeSpace.negative_profiles_zero(negative_array)
    #
    # Negative tests
    # def test_subtract_baseline_bad_data(self):
    #     with self.assertRaises(Exception):
    #         warnings.filterwarnings("ignore")
    #         TimeSpace.get_indata_txt('resources/empty.dat', '')
    # -------------------------------------------------------------------

    # Write test for calculate vself

if __name__ == '__main__':
    unittest.main()
