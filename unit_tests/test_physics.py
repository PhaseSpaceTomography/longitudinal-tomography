import unittest
from Physics import *
from Parameters import Parameters

# In this test-suite i am testing individual physics formulas,
#  mostly using made-up values


class TestPhysics(unittest.TestCase):

    # Run before every test
    def setUp(self):
        self.par = Parameters()

    def test_b_to_e_function(self):
        self.par.q = 1.0
        self.par.b0 = 0.5
        self.par.bending_rad = 8.1
        self.par.e_rest = 0.93827231E9
        self.assertAlmostEqual(b_to_e(self.par), 1534450425.929688,
                               msg="Error in calculation of B to E")

    def test_lorentz_beta_function(self):
        self.par.e_rest = 0.93827231E9
        self.par.e0 = np.array([0.93995544E9])
        self.assertAlmostEqual(lorenz_beta(self.par, rf_turn=0),
                               0.05981714641718171,
                               msg="Error in calculation of lorenz_beta")

    def test_rfvolt_rf1_function(self):
        self.par.time_at_turn = np.array([0.0000245])
        self.par.vrf1 = 7945.4036
        self.par.vrf1dot = 37.556
        self.par.q = 1.0
        self.par.mean_orbit_rad = 25.0
        self.par.bending_rad = 8.239
        self.par.bdot = 1.88365
        phi = 0.23456
        self.assertAlmostEqual(rfvolt_rf1(phi, self.par, rf_turn=0)[0],
                               -591.1488078823318,
                               msg="Error in calculation of rfvolt_rf1")

    def test_drfvolt_rf1_function(self):
        self.par.time_at_turn = np.array([0.000224563])
        self.par.vrf1 = 7945.4036
        self.par.vrf1dot = 37.556
        phi = 0.012463
        self.assertAlmostEqual(drfvolt_rf1(phi, self.par, rf_turn=0),
                               7944.794975674504,
                               msg="Error in calculation of drfvolt_rf1")

    def test_rf_voltage_function(self):
        self.par.time_at_turn = np.array([0.000224563])
        self.par.vrf1 = 7945.4036
        self.par.vrf1dot = 37.556
        self.par.vrf2 = -1245.2236
        self.par.vrf2dot = 5.5136
        self.par.hratio = 16
        self.par.phi12 = np.array([0.0002354])
        self.par.mean_orbit_rad = 23.1
        self.par.bdot = 2.31475
        self.par.bending_rad = 9.123
        self.par.q = 1.0
        phi = 0.02342
        self.assertAlmostEqual(rf_voltage(phi, self.par, 0)[0],
                               -2878.957599969201,
                               msg="Error in calculation of rf_voltage")

    def test_drf_voltage(self):
        self.par.time_at_turn = np.array([0.0042])
        self.par.vrf1 = 7945.4036
        self.par.vrf1dot = 37.556
        self.par.vrf2 = -1245.2236
        self.par.vrf2dot = 5.5136
        self.h_ratio = 16
        self.par.phi12 = np.array([0.1214])
        phi = 0.5323
        self.assertAlmostEqual(drf_voltage(
                                   phi, parameters=self.par, rf_turn=0)[0],
                               6846.229858420835,
                               msg="Error in calculation of drf_voltage")

    def test_short_rf_voltage_formula(self):
        phi = 0.0234
        vrf1 = 7945.4036
        vrf1dot = 37.556
        vrf2 = -1245.2236
        vrf2dot = 5.5136
        h_ratio = 12
        phi12 = 0.124
        time_at_turn = np.array([0.02321])
        rf_turn = 0
        self.assertAlmostEqual(short_rf_voltage_formula(
                                    phi, vrf1, vrf1dot,
                                    vrf2, vrf2dot, h_ratio,
                                    phi12, time_at_turn, rf_turn),
                               1349.6219819508497,
                               msg="Error in calculation of "
                                   "short_rf_voltage_formula")

    def test_vrft(self):
        vrf1 = 7945.4036
        vrf1dot = 37.556
        turn_time = 0.001223
        self.assertAlmostEqual(vrft(vrf1, vrf1dot, turn_time),
                               7945.449530987999,
                               msg="Error in calculation of vrft")

    def test_phi_lower_upper_function_condition_true(self):
        self.par.q = 1.0
        self.par.trans_gamma = 4.1
        self.par.e0 = np.array([0.93995544E9])
        self.par.e_rest = 0.93827231E9
        phi_lower, phi_upper = find_phi_lower_upper(self.par, rf_turn=0)
        self.assertAlmostEqual(phi_lower, -1.0 * np.pi,
                               msg="Error in calculation of phi_lower")
        self.assertAlmostEqual(phi_upper, np.pi,
                               msg="Error in calculation of phi_upper")

    def test_phi_lower_upper_condition_true(self):
        self.par.q = -1.0
        self.par.trans_gamma = 4.1
        self.par.e0 = np.array([0.93995544E9])
        self.par.e_rest = 0.93827231E9
        phi_lower, phi_upper = find_phi_lower_upper(self.par, rf_turn=0)
        self.assertAlmostEqual(phi_lower, 0.0,
                               msg="Error in calculation of phi_lower")
        self.assertAlmostEqual(phi_upper, 2 * np.pi,
                               msg="Error in calculation of phi_upper")

    def test_find_phase_slip_factor(self):
        self.par.beta0 = np.array([0.7235633223])
        self.par.trans_gamma = 4.1
        self.assertAlmostEqual(phase_slip_factor(self.par)[0],
                               0.41696771886014,
                               msg="Error in calculation of "
                                   "phase_slip_factor")

    def test_find_c1(self):
        self.par.h_num = 12
        self.par.eta0 = np.array([0.45672345])
        self.par.e0 = np.array([0.93995544E9])
        self.par.beta0 = np.array([0.7235633223])
        self.assertAlmostEqual(find_c1(self.par)[0],
                               6.997679876644275e-08,
                               msg="Error in calculation of c1")

    def test_revolution_freq(self):
        self.par.beta0 = np.array([0.7235633223])
        self.par.mean_orbit_rad = 25.0
        self.assertAlmostEqual(revolution_freq(self.par)[0],
                               8676753.076438528,
                               msg="Error in calculation of revolution_freq")

    def test_calc_self_field_coeffs(self):
        self.par.profile_count = 1
        self.par.dturns = 12
        self.par.omega_rev0 = np.array([8676753.076438528])
        self.par.beta0 = np.array([0.7235633223])
        self.par.g_coupling = 0.01
        self.par.zwall_over_n = 0.01
        self.par.dtbin = 5.0E10
        self.assertAlmostEqual(calc_self_field_coeffs(self.par)[0],
                               9.087506098118859e-48,
                               msg="Error in calculation of "
                                   "self_field_coeffs")

    def test_phase_low(self):
        self.par.vrf1 = 7945.4036
        self.par.vrf1dot = 37.556
        self.par.vrf2 = -1245.2236
        self.par.vrf2dot = 5.5136
        self.par.h_ratio = 12
        self.par.bunch_phaselength = 100
        self.par.phi0 = np.array([0.00123])
        self.par.phi12 = 0.124
        self.par.time_at_turn = np.array([0.00121])
        phase = 0.0023213
        self.assertAlmostEqual(phase_low(phase, self.par, rf_turn=0),
                               123831.46073450413,
                               msg="Error in calculation of phase_low")

    def test_dphase_low(self):
        self.par.vrf1 = 7945.4036
        self.par.vrf2 = -1245.2236
        self.par.h_ratio = 12
        self.par.bunch_phaselength = 100
        self.par.phi12 = 0.124
        phase = 0.0023213
        self.assertAlmostEqual(dphase_low(phase, self.par, rf_turn=0),
                               4018.5009484603393,
                               msg="Error in calculation of "
                                   "derivative of phase_low")

    def test_find_synch_phase(self):
        self.par.vrf1 = 7945.4036
        self.par.vrf1dot = 37.556
        self.par.vrf2 = -1245.2236
        self.par.vrf2dot = 5.5136
        self.par.h_ratio = 12
        self.par.bunch_phaselength = 100
        self.par.phi0 = np.array([0.00123])
        self.par.phi12 = 0.124
        self.par.time_at_turn = np.array([0.00121])
        phi_lower = -1.0 * np.pi
        phi_upper = np.pi
        rf_turn = 0
        self.assertAlmostEqual(find_synch_phase(
                                   self.par, rf_turn, phi_lower, phi_upper),
                               -0.08811487062442194,
                               msg="Error in calculation of synchronous phase")


if __name__ == '__main__':

    unittest.main()
