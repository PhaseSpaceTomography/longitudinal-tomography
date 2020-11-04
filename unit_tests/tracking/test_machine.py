"""Unit-tests for the Machine class.

Run as python test_machine.py in console or via coverage
"""

import unittest

import tomo.tracking.machine as mch


# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = {
    'output_dir':          '/tmp/',
    'dtbin':               9.999999999999999E-10,
    'dturns':              5,
    'synch_part_x':        334.00000000000006,
    'demax': -1.E6,
    'filmstart':           0,
    'filmstop':            1,
    'filmstep':            1,
    'niter':               20,
    'snpt':                4,
    'full_pp_flag':        False,
    'beam_ref_frame':      0,
    'machine_ref_frame':   0,
    'vrf1':                2637.197030932989,
    'vrf1dot':             0.0,
    'vrf2':                0.0,
    'vrf2dot':             0.0,
    'h_num':               1,
    'h_ratio':             2.0,
    'phi12':               0.4007821253666541,
    'b0':                  0.15722,
    'bdot':                0.7949999999999925,
    'mean_orbit_rad':      25.0,
    'bending_rad':         8.239,
    'trans_gamma':         4.1,
    'rest_energy':         0.93827231E9,
    'charge':              1,
    'self_field_flag':     False,
    'g_coupling':          0.0,
    'zwall_over_n':        0.0,
    'pickup_sensitivity':  0.36,
    'nprofiles':           150,
    'nbins':               760,
    'min_dt':              0.0,
    'max_dt':              9.999999999999999E-10 * 760  # dtbin * nbins
}


class TestMachine(unittest.TestCase):

    def test_set_nbins_correct_nbins(self):
        new_nbins = 200
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nbins = new_nbins
        self.assertEqual(machine.nbins, new_nbins,
                         msg='nbins was set incorrectly')

    def test_set_nbins_correct_yat0(self):
        new_nbins = 200
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nbins = new_nbins
        self.assertEqual(machine.synch_part_y, 100.0,
                         msg='yat0 was set incorrectly when nbins was updated')

    def test_values_at_turns_correct_length(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        self.assertEqual(len(machine.time_at_turn), 746,
                         msg='Wrong length of array: time_at_turn')
        self.assertEqual(len(machine.omega_rev0), 746,
                         msg='Wrong length of array: omega_rev0')
        self.assertEqual(len(machine.phi0), 746,
                         msg='Wrong length of array: phi0')
        self.assertEqual(len(machine.drift_coef), 746,
                         msg='Wrong length of array: drift_coef')
        self.assertEqual(len(machine.deltaE0), 746,
                         msg='Wrong length of array: deltaE0')
        self.assertEqual(len(machine.beta0), 746,
                         msg='Wrong length of array: beta0')
        self.assertEqual(len(machine.eta0), 746,
                         msg='Wrong length of array: eta0')
        self.assertEqual(len(machine.e0), 746,
                         msg='Wrong length of array: e0')
        self.assertEqual(len(machine.vrf1_at_turn), 746,
                         msg='Wrong length of array: vrf1_at_turn')
        self.assertEqual(len(machine.vrf2_at_turn), 746,
                         msg='Wrong length of array: vrf2_at_turn')

    def test_values_at_turns_correct_time_at_turn(self):
        nprofs = 5
        machine_ref_frame = 2

        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = nprofs
        machine.machine_ref_frame = machine_ref_frame
        machine.values_at_turns()

        correct = [-1.37016417e-05, -1.23314411e-05, -1.09612485e-05,
                   -9.59106409e-06, -8.22088776e-06, -6.85071954e-06,
                   -5.48055942e-06, -4.11040741e-06, -2.74026350e-06,
                   -1.37012770e-06, 0.00000000e+00, 1.37011959e-06,
                   2.74023109e-06, 4.11033447e-06, 5.48042976e-06,
                   6.85051693e-06, 8.22059601e-06, 9.59066698e-06,
                   1.09607299e-05, 1.23307846e-05, 1.37008313e-05]

        for tat, corr in zip(machine.time_at_turn, correct):
            self.assertAlmostEqual(tat, corr,
                                   msg='Error in calculation of time at turn')

    def test_values_at_turns_correct_omega_rev0(self):
        nprofs = 5
        machine_ref_frame = 2

        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = nprofs
        machine.machine_ref_frame = machine_ref_frame
        machine.values_at_turns()

        correct = [4585595.05834437, 4585622.18581794, 4585649.31304857,
                   4585676.44003627, 4585703.56678104, 4585730.69328289,
                   4585757.81954182, 4585784.94555784, 4585812.07133096,
                   4585839.19686116, 4585866.32214847, 4585893.44719289,
                   4585920.57199441, 4585947.69655305, 4585974.82086881,
                   4586001.94494169, 4586029.0687717,  4586056.19235885,
                   4586083.31570313, 4586110.43880456, 4586137.56166313]

        for omega, corr in zip(machine.omega_rev0, correct):
            self.assertAlmostEqual(omega, corr,
                                   msg='Error in calculation of revolution '
                                       'frequency (omega_rev0)')

    def test_values_at_turns_correct_phi0(self):
        nprofs = 5
        machine_ref_frame = 2

        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = nprofs
        machine.machine_ref_frame = machine_ref_frame
        machine.values_at_turns()

        correct = [0.40078213, 0.40078213, 0.40078213, 0.40078213,
                   0.40078213, 0.40078213, 0.40078213, 0.40078213,
                   0.40078213, 0.40078213, 0.40078213, 0.40078213,
                   0.40078213, 0.40078213, 0.40078213, 0.40078213,
                   0.40078213, 0.40078213, 0.40078213, 0.40078213,
                   0.40078213]

        for phi, corr in zip(machine.phi0, correct):
            self.assertAlmostEqual(phi, corr,
                                   msg='Error in calculation of synchronous '
                                       'phase at each turn (phi0)')

    def test_values_at_turns_correct_drift_coef(self):
        nprofs = 5
        machine_ref_frame = 2

        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = nprofs
        machine.machine_ref_frame = machine_ref_frame
        machine.values_at_turns()

        correct = [3.36099331e-08, 3.36094281e-08, 3.36089232e-08,
                   3.36084183e-08, 3.36079135e-08, 3.36074086e-08,
                   3.36069038e-08, 3.36063989e-08, 3.36058941e-08,
                   3.36053893e-08, 3.36048845e-08, 3.36043797e-08,
                   3.36038749e-08, 3.36033702e-08, 3.36028654e-08,
                   3.36023607e-08, 3.36018560e-08, 3.36013513e-08,
                   3.36008466e-08, 3.36003419e-08, 3.35998373e-08]

        for drift, corr in zip(machine.drift_coef, correct):
            self.assertAlmostEqual(drift, corr,
                                   msg='Error in calculation of drift '
                                       'coefficient (drift_coef)')

    def test_values_at_turns_correct_deltaE0(self):
        nprofs = 5
        machine_ref_frame = 2

        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = nprofs
        machine.machine_ref_frame = machine_ref_frame
        machine.values_at_turns()

        correct = [1028.87237942, 1028.87237942, 1028.87237942, 1028.87237942,
                   1028.87237942, 1028.87237942, 1028.87237942, 1028.87237942,
                   1028.87237942, 1028.87237942,    0.,         1028.87237942,
                   1028.87237942, 1028.87237942, 1028.87237942, 1028.87237942,
                   1028.87237942, 1028.87237942, 1028.87237942, 1028.87237942,
                   1028.87237942]

        for dE0, corr in zip(machine.deltaE0, correct):
            self.assertAlmostEqual(dE0, corr,
                                   msg='Error in calculation of energy '
                                       'difference of synch part pr turn '
                                       '(deltaE0)')

    def test_values_at_turns_correct_beta0(self):
        nprofs = 5
        machine_ref_frame = 2

        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = nprofs
        machine.machine_ref_frame = machine_ref_frame
        machine.values_at_turns()

        correct = [0.38239747, 0.38239973, 0.38240199, 0.38240425, 0.38240652,
                   0.38240878, 0.38241104, 0.3824133,  0.38241556, 0.38241783,
                   0.38242009, 0.38242235, 0.38242461, 0.38242687, 0.38242914,
                   0.3824314,  0.38243366, 0.38243592, 0.38243818, 0.38244044,
                   0.38244271]

        for beta, corr in zip(machine.beta0, correct):
            self.assertAlmostEqual(beta, corr,
                                   msg='Error in calculation of relativistic '
                                       'beta (beta0)')

    def test_values_at_turns_correct_eta0(self):
        nprofs = 5
        machine_ref_frame = 2

        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = nprofs
        machine.machine_ref_frame = machine_ref_frame
        machine.values_at_turns()

        correct = [0.79428378, 0.79428205, 0.79428032, 0.79427859, 0.79427686,
                   0.79427513, 0.7942734,  0.79427167, 0.79426994, 0.79426821,
                   0.79426648, 0.79426475, 0.79426302, 0.79426129, 0.79425956,
                   0.79425783, 0.7942561,  0.79425437, 0.79425264, 0.79425091,
                   0.79424918]

        for eta, corr in zip(machine.eta0, correct):
            self.assertAlmostEqual(eta, corr,
                                   msg='Error in calculation of phase slip '
                                       'factor (eta0)')

    # This array is tested as integers due to its high values.
    def test_values_at_turns_correct_e0(self):
        nprofs = 5
        machine_ref_frame = 2

        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = nprofs
        machine.machine_ref_frame = machine_ref_frame
        machine.values_at_turns()

        correct = [1015448496, 1015449524, 1015450553, 1015451582, 1015452611,
                   1015453640, 1015454669, 1015455698, 1015456727, 1015457755,
                   1015458784, 1015459813, 1015460842, 1015461871, 1015462900,
                   1015463929, 1015464958, 1015465986, 1015467015, 1015468044,
                   1015469073]

        for e0, corr in zip(machine.e0, correct):
            self.assertEqual(int(e0), corr,
                             msg='Error in calculation of energy '
                                 'of synch. particle (e0)')

    def test_values_at_turns_correct_vrf_with_derivative(self):
        nprofs = 5
        machine_ref_frame = 2

        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = nprofs
        machine.machine_ref_frame = machine_ref_frame
        machine.vrf1dot = 10
        machine.values_at_turns()

        correct = [2637.19689392, 2637.19690762, 2637.19692132, 2637.19693502,
                   2637.19694872, 2637.19696243, 2637.19697613, 2637.19698983,
                   2637.19700353, 2637.19701723, 2637.19703093, 2637.19704463,
                   2637.19705834, 2637.19707204, 2637.19708574, 2637.19709944,
                   2637.19711314, 2637.19712684, 2637.19714054, 2637.19715424,
                   2637.19716794]

        for vrf, corr in zip(machine.vrf1_at_turn, correct):
            self.assertAlmostEqual(vrf, corr,
                                   msg='Error in calculation of RF voltage '
                                       'with vrf1dot (vrf1_at_turn)')
