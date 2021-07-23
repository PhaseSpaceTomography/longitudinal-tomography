"""Unit-tests for the physics module.

Run as python test_physics.py in console or via coverage
"""

import unittest

import numpy as np

from .. import commons
import longitudinal_tomography.tracking.machine as mch
import longitudinal_tomography.utils.physics as physics

# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = commons.get_machine_args()


class TestPhysics(unittest.TestCase):

    def test_b_to_e_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        correct = 1015458784.835785
        ans = physics.b_to_e(machine)
        self.assertAlmostEqual(
            ans, correct, msg='Error in energy calculated from B field')

    def test_lorenz_beta_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.e0 = [1015458784.835785]
        correct = 0.382420087611783
        ans = physics.lorentz_beta(machine, rf_turn=0)
        self.assertAlmostEqual(
            ans, correct, msg='Error in calculation of Lorentz beta')

    def test_rfvolt_rf1_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.time_at_turn = [1.3701195948153858e-06]
        correct = -705.3144383117857
        ans = physics.rfvolt_rf1_mch(0.123, machine, rf_turn=0)

        self.assertAlmostEqual(
            float(ans), correct, msg='Error in calculation of rfvolt1')

    def test_drfvolt_rf1_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.time_at_turn = [1.3701195948153858e-06]
        correct = 2617.2730921111274
        ans = physics.drfvolt_rf1_mch(0.123, machine, rf_turn=0)

        self.assertAlmostEqual(
            float(ans), correct, msg='Error in calculation drfvolt1')

    def test_rf_voltage_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.vrf2 = 1250
        machine.time_at_turn = [1.3701195948153858e-06]
        correct = -1364.5929048626685
        ans = physics.rf_voltage_mch(0.123, machine, rf_turn=0)

        self.assertAlmostEqual(
            ans, correct, msg='Error in calculation rf_voltage')

    def test_drf_voltage_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.vrf2 = 1250
        machine.time_at_turn = [1.3701195948153858e-06]
        correct = 4741.280534228331
        ans = physics.drf_voltage_mch(0.123, machine, rf_turn=0)

        self.assertAlmostEqual(
            ans, correct, msg='Error in calculation drf_voltage')

    def test_rf_voltage_correct2(self):
        machine = mch.Machine(**MACHINE_ARGS)
        correct = -1132.2371121228516
        ans = physics.rf_voltage_at_phase(
            0.123, 2500, 0, 2135, 0,
            2, 1.324, [1.3701195948],
            0)

        self.assertAlmostEqual(
            ans, correct, msg='Error in calculation drf_voltage')

    def test_vrft_correct(self):
        correct = 1500
        ans = physics.vrft(500, 250, 4)
        self.assertEqual(ans, correct, msg='Error in calculation vrft')

    def test_find_synchronous_phase_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.time_at_turn = [1.3701195948153858e-06]
        machine.e0 = [1015458784.835785]

        correct = 2.740810528223139
        ans = physics.find_synch_phase_mch(
            machine, rf_turn=0, phi_lower=0, phi_upper=2*np.pi)

        self.assertAlmostEqual(
            ans, correct, msg='Error in found synchronous phase')

    def test_find_phi_lower_upper_positive_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.e0 = [1015458784.835785]
        correct_low = -np.pi
        correct_up = np.pi
        ans_low, ans_up = physics.find_phi_lower_upper(machine, rf_turn=0)

        self.assertAlmostEqual(
            ans_low, correct_low, msg='Error in found calculation '
                                      'of lower phase')
        self.assertAlmostEqual(
            ans_up, correct_up, msg='Error in found calculation '
                                    'of upper phase')

    def test_find_phi_lower_upper_negative_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.e0 = [1015458784.835785]
        machine.q *= -1
        correct_low = 0.0
        correct_up = 2 * np.pi
        ans_low, ans_up = physics.find_phi_lower_upper(machine, rf_turn=0)

        self.assertAlmostEqual(
            ans_low, correct_low, msg='Error in found calculation '
                                      'of lower phase')
        self.assertAlmostEqual(
            ans_up, correct_up, msg='Error in found calculation '
                                    'of upper phase')

    def test_phase_slip_factor_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.beta0 = np.array([0.38242009])
        correct = 0.7942664750023455
        ans = physics.phase_slip_factor(machine)
        ans = ans[0]
        self.assertAlmostEqual(
            ans, correct, msg='Error in found calculation phase slip factor')

    def test_find_dphase_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.beta0 = np.array([0.38242009])
        machine.eta0 = np.array([0.79426648])
        machine.e0 = np.array([1015458784.835785])

        correct = 3.360488420030597e-08
        ans = physics.find_dphase(machine)
        ans = ans[0]

        self.assertAlmostEqual(
            ans, correct, msg='Error in found calculation dphase')

    def test_calc_revolution_freq_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.beta0 = np.array([0.38242009])

        correct = 4585866.350787248
        ans = physics.revolution_freq(machine)
        ans = ans[0]
        self.assertAlmostEqual(
            ans, correct, msg='Error in found calculation of '
                              'revolution frequency')

    def test_calc_self_field_coeffs_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.beta0 = np.array([0.38242009])
        machine.omega_rev0 = np.array([4585866.32214847])
        machine.nprofiles = 1
        machine.dturns = 1
        machine.g_coupling = 1.0
        machine.zwall_over_n = 50.0

        correct = 1.2945175200855999e-05
        ans = physics.calc_self_field_coeffs(machine)
        ans = ans[0]

        self.assertAlmostEqual(
            ans, correct, msg='Error in found calculation of '
                              'self-field coefficients')

    def test_phase_low_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.phi0 = [0.40078213]
        machine.time_at_turn = [1.3701195948153858e-06]
        bunch_phaselength = 3.22312485522767
        correct = -1883.316491755506
        ans = physics.phase_low_mch(0.123, machine, bunch_phaselength,
                                    rf_turn=0)

        self.assertAlmostEqual(ans, correct,
                               msg='Phase_low calculated incorrectly')

    def test_dphase_low_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.phi0 = [0.40078213]
        machine.time_at_turn = [1.3701195948153858e-06]
        bunch_phaselength = 3.22312485522767
        correct = 859.1967476555656
        ans = physics.dphase_low_mch(0.123, machine, bunch_phaselength, 0,
                                     None)

        self.assertAlmostEqual(ans, correct,
                               msg='dphase_low calculated incorrectly')
