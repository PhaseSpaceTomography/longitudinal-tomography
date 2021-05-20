"""Unit-tests for the functions in the tomolib_wrappers module.

Run as python test_tomolib_wrappers.py in console or via coverage
"""

import os
import unittest

import numpy as np
import numpy.testing as nptest

from .. import commons
import longitudinal_tomography.tracking.machine as mch
from longitudinal_tomography.cpp_routines import libtomo

# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = commons.get_machine_args()


class TestTLW(unittest.TestCase):

    def test_kick_up_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 10
        machine.vrf2 = 500
        machine.values_at_turns()

        denergy = np.array([-115567.32591061])
        dphi = np.array([0.33178332])
        npart = 1
        turn = 0

        rfv1 = machine.vrf1_at_turn * machine.q
        rfv2 = machine.vrf2_at_turn * machine.q

        new_denergy = libtomo.kick(machine, denergy, dphi,
                                   rfv1, rfv2, npart, turn, up=True)

        correct_energy = -113495.65825924404
        self.assertAlmostEqual(
            new_denergy[0], correct_energy,
            msg='Kick in upward direction was calculated incorrectly')

    def test_kick_down_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 10
        machine.vrf2 = 500
        machine.values_at_turns()

        denergy = np.array([-115567.32591061])
        dphi = np.array([0.33178332])
        npart = 1
        turn = (machine.nprofiles - 1) * machine.dturns

        rfv1 = machine.vrf1_at_turn * machine.q
        rfv2 = machine.vrf2_at_turn * machine.q

        new_denergy = libtomo.kick(machine, denergy, dphi,
                                   rfv1, rfv2, npart, turn, False)

        correct_energy = -116610.12118255378
        self.assertAlmostEqual(
            new_denergy[0], correct_energy,
            msg='Kick in downward direction was calculated incorrectly')

    def test_drift_up_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 10
        machine.values_at_turns()

        denergy = np.array([-115567.32591061])
        dphi = np.array([0.33178332])
        nparts = 1
        turn = 0

        new_dphi = libtomo.drift(denergy, dphi, machine.drift_coef,
                                 nparts, turn, up=True)

        correct_dphi = 0.3356669466375665
        self.assertAlmostEqual(
            new_dphi[0], correct_dphi,
            msg='Drift in upward direction was calculated incorrectly')

    def test_drift_down_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 10
        machine.values_at_turns()

        denergy = np.array([-115567.32591061])
        dphi = np.array([0.33178332])
        nparts = 1
        turn = (machine.nprofiles - 1) * machine.dturns

        new_dphi = libtomo.drift(denergy, dphi, machine.drift_coef,
                                 nparts, turn, up=False)

        correct_dphi = 0.3279023169434031
        self.assertAlmostEqual(
            new_dphi[0], correct_dphi,
            msg='Drift in downward direction was calculated incorrectly')

    def test_kick_and_drift_machine_arg_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 10
        machine.values_at_turns()

        denergy = np.array([-115567.32591061])
        dphi = np.array([0.33178332])
        nparts = 1

        recprof = 5
        nturns = machine.dturns * (machine.nprofiles - 1)

        rfv1 = machine.vrf1_at_turn * machine.q
        rfv2 = machine.vrf2_at_turn * machine.q

        xp = np.zeros((machine.nprofiles, nparts))
        yp = np.zeros((machine.nprofiles, nparts))

        xp, yp = libtomo.kick_and_drift(
            xp, yp, denergy, dphi, rfv1, rfv2, machine, recprof,
            nturns, nparts)

        correct_xp = np.array([[0.22739336], [0.24930078], [0.27073013],
                               [0.291644], [0.31200651], [0.33178332],
                               [0.35094168], [0.36945041], [0.38727993],
                               [0.40440224]])
        correct_yp = np.array([[-131465.56602927], [-128721.12723285],
                               [-125750.07868119], [-122561.32895953],
                               [-119163.98960288], [-115567.32591061],
                               [-111780.71031054], [-107813.57867853],
                               [-103675.3899576], [-99375.58935766]])

        nptest.assert_almost_equal(xp, correct_xp,
                                   err_msg='Error in tracked x coordinates')

        nptest.assert_almost_equal(yp, correct_yp,
                                   err_msg='Error in tracked y coordinates')

    def test_kick_and_drift_args_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 10
        machine.values_at_turns()

        denergy = np.array([-115567.32591061])
        dphi = np.array([0.33178332])
        nparts = 1

        recprof = 5
        nturns = machine.dturns * (machine.nprofiles - 1)

        rfv1 = machine.vrf1_at_turn * machine.q
        rfv2 = machine.vrf2_at_turn * machine.q

        xp = np.zeros((machine.nprofiles, nparts))
        yp = np.zeros((machine.nprofiles, nparts))

        phi0 = machine.phi0
        deltaE0 = machine.deltaE0
        omega_rev0 = machine.omega_rev0
        drift_coef = machine.drift_coef
        phi12 = machine.phi12
        h_ratio = machine.h_ratio
        dturns = machine.dturns

        xp, yp = libtomo.kick_and_drift(
            xp, yp, denergy, dphi, rfv1, rfv2, phi0, deltaE0, drift_coef,
            phi12, h_ratio, dturns, recprof, nturns, nparts, False)

        correct_xp = np.array([[0.22739336], [0.24930078], [0.27073013],
                               [0.291644], [0.31200651], [0.33178332],
                               [0.35094168], [0.36945041], [0.38727993],
                               [0.40440224]])
        correct_yp = np.array([[-131465.56602927], [-128721.12723285],
                               [-125750.07868119], [-122561.32895953],
                               [-119163.98960288], [-115567.32591061],
                               [-111780.71031054], [-107813.57867853],
                               [-103675.3899576], [-99375.58935766]])

        nptest.assert_almost_equal(xp, correct_xp,
                                   err_msg='Error in tracked x coordinates')

        nptest.assert_almost_equal(yp, correct_yp,
                                   err_msg='Error in tracked y coordinates')

    def test_kick_and_drift_few_args_fails(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 10
        machine.values_at_turns()

        denergy = np.array([-115567.32591061])
        dphi = np.array([0.33178332])
        nparts = 1

        recprof = 5
        nturns = machine.dturns * (machine.nprofiles - 1)

        rfv1 = machine.vrf1_at_turn * machine.q
        rfv2 = machine.vrf2_at_turn * machine.q

        xp = np.zeros((machine.nprofiles, nparts))
        yp = np.zeros((machine.nprofiles, nparts))

        phi0 = machine.phi0
        deltaE0 = machine.deltaE0
        omega_rev0 = machine.omega_rev0
        drift_coef = machine.drift_coef
        phi12 = machine.phi12
        h_ratio = machine.h_ratio

        with self.assertRaises(TypeError,
                               msg='Too few arrays should '
                                   'raise an exception'):
            xp, yp = libtomo.kick_and_drift(
                xp, yp, denergy, dphi, rfv1, rfv2, phi0, deltaE0, drift_coef,
                phi12, h_ratio, recprof, nturns, nparts)

        dturns = machine.dturns
        some_useless_var = None
        with self.assertRaises(TypeError,
                               msg='Too many arrays should '
                                   'raise an exception'):
            xp, yp = libtomo.kick_and_drift(
                xp, yp, denergy, dphi, rfv1, rfv2, phi0, deltaE0, drift_coef,
                phi12, h_ratio, some_useless_var, dturns, recprof, nturns,
                nparts)

    def test_back_project(self):
        waterfall = self._load_waterfall()

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        nbins = len(waterfall[0])

        weights = np.zeros(nparts, dtype=np.float64)

        flat_profs = np.ascontiguousarray(
            waterfall.flatten()).astype(np.float64)

        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]
        xp = xp.T
        flat_points = xp.copy()
        for i in range(nprofs):
            flat_points[:, i] += nbins * i
        flat_points = np.ascontiguousarray(flat_points).astype(np.int32)

        libtomo.back_project(weights, flat_points, flat_profs, nparts, nprofs)

        cweights = np.array([0.79054276, 0.80812089, 0.81983964, 0.83468339,
                             0.84327714, 0.84679276, 0.85187089, 0.85187089,
                             0.85148026, 0.85304276, 0.85304276, 0.84601151,
                             0.84444901, 0.84249589, 0.84054276, 0.83937089,
                             0.84405839, 0.83780839, 0.84249589, 0.84796464,
                             0.84483964, 0.85421464, 0.86319901, 0.87530839,
                             0.88038651, 0.88976151, 0.90030839, 0.90733964,
                             0.91671464, 0.92491776, 0.93077714, 0.93937089,
                             0.93780839, 0.94366776, 0.94210526, 0.93780839,
                             0.93312089, 0.93194901, 0.92804276, 0.92530839,
                             0.91749589, 0.91124589, 0.90968339, 0.90421464,
                             0.89718339, 0.89132401, 0.89171464, 0.88780839,
                             0.88273026, 0.88273026])

        nptest.assert_almost_equal(
            weights, cweights, err_msg='Weights were calculated incorrectly')

    def test_project_correct(self):
        nprofs = 10
        nparts = 50
        nbins = 100

        waterfall_shape = (nprofs, nbins)

        weights = np.array([0.79054276, 0.80812089, 0.81983964, 0.83468339,
                            0.84327714, 0.84679276, 0.85187089, 0.85187089,
                            0.85148026, 0.85304276, 0.85304276, 0.84601151,
                            0.84444901, 0.84249589, 0.84054276, 0.83937089,
                            0.84405839, 0.83780839, 0.84249589, 0.84796464,
                            0.84483964, 0.85421464, 0.86319901, 0.87530839,
                            0.88038651, 0.88976151, 0.90030839, 0.90733964,
                            0.91671464, 0.92491776, 0.93077714, 0.93937089,
                            0.93780839, 0.94366776, 0.94210526, 0.93780839,
                            0.93312089, 0.93194901, 0.92804276, 0.92530839,
                            0.91749589, 0.91124589, 0.90968339, 0.90421464,
                            0.89718339, 0.89132401, 0.89171464, 0.88780839,
                            0.88273026, 0.88273026], dtype=np.float64)

        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]
        xp = xp.T
        flat_points = xp.copy()
        for i in range(nprofs):
            flat_points[:, i] += nbins * i
        flat_points = np.ascontiguousarray(flat_points).astype(np.int32)

        rec = libtomo.project(np.zeros(waterfall_shape), flat_points,
                              weights, nparts, nprofs, nbins)

        correct = [0.79054276, 0.80812089, 0.81983964, 0.83468339,
                   0.84327714, 0.84679276, 0.85187089, 0.85187089,
                   0.85148026, 0.85304276, 0.85304276, 0.84601151,
                   0.84444901, 0.84249589, 0.84054276, 0.83937089,
                   0.84405839, 0.83780839, 0.84249589, 0.84796464,
                   0.84483964, 0.85421464, 0.86319901, 0.87530839,
                   0.88038651, 0.88976151, 0.90030839, 0.90733964,
                   0.91671464, 0.92491776, 0.93077714, 0.93937089,
                   0.93780839, 0.94366776, 0.94210526, 0.93780839,
                   0.93312089, 0.93194901, 0.92804276, 0.92530839,
                   0.91749589, 0.91124589, 0.90968339, 0.90421464,
                   0.89718339, 0.89132401, 0.89171464, 0.88780839,
                   0.88273026, 0.88273026, 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0.]

        correct = np.array([correct] * 10)

        nptest.assert_almost_equal(
            rec, correct,
            err_msg='Error in recreated profile data after projection')

    def test_old_reconstruct_correct(self):
        nprofs = 10
        nparts = 50
        nbins = 100
        niter = 1

        weights = np.zeros(nparts)
        discr = np.zeros(niter+1)

        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]
        xp = xp.T

        xp = np.ascontiguousarray(xp).astype(np.int32)

        waterfall = self._load_waterfall()
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        flat_profs = np.ascontiguousarray(
            waterfall.flatten()).astype(np.float64)

        weights, discr = libtomo.reconstruct_old(
            weights, xp, flat_profs, discr, niter,
            nbins, nparts, nprofs, verbose=False)

        correct_w = np.array([1.40130575, 1.4324645, 1.45323701, 1.47954884,
                              1.49478201, 1.50101376, 1.51001518, 1.51001518,
                              1.50932276, 1.51209243, 1.51209243, 1.49962893,
                              1.49685926, 1.49339718, 1.48993509, 1.48785784,
                              1.49616684, 1.48508818, 1.49339718, 1.50309101,
                              1.49755168, 1.51416968, 1.53009527, 1.55156019,
                              1.5605616,  1.57717961, 1.59587486, 1.60833836,
                              1.62495636, 1.63949711, 1.64988337, 1.66511653,
                              1.66234687, 1.67273312, 1.66996345, 1.66234687,
                              1.65403787, 1.65196062, 1.64503645, 1.64018953,
                              1.6263412,  1.61526253, 1.61249286, 1.60279903,
                              1.59033552, 1.57994927, 1.58064169, 1.57371752,
                              1.5647161,  1.5647161])

        correct_discr = np.array([0.0745208, 0.0745208])

        nptest.assert_almost_equal(
            weights, correct_w,
            err_msg='Error in weights after reconstruction.')

        nptest.assert_almost_equal(
            discr, correct_discr,
            err_msg='Error in calculated discrepancies after reconstruction.')

    def test_reconstruct_correct(self):
        nprofs = 10
        nparts = 50
        nbins = 100
        niter = 1

        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]
        xp = xp.T

        waterfall = self._load_waterfall()
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]

        (weights,
         discr,
         recreated) = libtomo.reconstruct(
            xp, waterfall, niter,
            nbins, nparts, nprofs,
            verbose=False)

        correct_w = np.array([1.40130575, 1.4324645,  1.45323701, 1.47954884,
                              1.49478201, 1.50101376, 1.51001518, 1.51001518,
                              1.50932276, 1.51209243, 1.51209243, 1.49962893,
                              1.49685926, 1.49339718, 1.48993509, 1.48785784,
                              1.49616684, 1.48508818, 1.49339718, 1.50309101,
                              1.49755168, 1.51416968, 1.53009527, 1.55156019,
                              1.5605616,  1.57717961, 1.59587486, 1.60833836,
                              1.62495636, 1.63949711, 1.64988337, 1.66511653,
                              1.66234687, 1.67273312, 1.66996345, 1.66234687,
                              1.65403787, 1.65196062, 1.64503645, 1.64018953,
                              1.6263412,  1.61526253, 1.61249286, 1.60279903,
                              1.59033552, 1.57994927, 1.58064169, 1.57371752,
                              1.5647161,  1.5647161])

        correct_discr = np.array([0.0745208, 0.0745208])

        correct_rec = np.array([0.01797798, 0.01837773, 0.01864423, 0.01898179,
                                0.01917723, 0.01925718, 0.01937266, 0.01937266,
                                0.01936378, 0.01939931, 0.01939931, 0.01923941,
                                0.01920388, 0.01915946, 0.01911504, 0.01908839,
                                0.01919499, 0.01905286, 0.01915946, 0.01928383,
                                0.01921276, 0.01942596, 0.01963028, 0.01990566,
                                0.02002114, 0.02023434, 0.02047419, 0.02063409,
                                0.02084729, 0.02103384, 0.02116709, 0.02136252,
                                0.02132699, 0.02146024, 0.02142471, 0.02132699,
                                0.02122039, 0.02119374, 0.02110491, 0.02104272,
                                0.02086506, 0.02072292, 0.02068739, 0.02056303,
                                0.02040313, 0.02026988, 0.02027876, 0.02018993,
                                0.02007444, 0.02007444, 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0.])

        nptest.assert_almost_equal(
            weights, correct_w,
            err_msg='Error in weights after reconstruction.')

        nptest.assert_almost_equal(
            discr, correct_discr,
            err_msg='Error in calculated discrepancies after reconstruction.')

        for i in range(nprofs):
            nptest.assert_almost_equal(recreated[i], correct_rec,
                                       err_msg='Error in '
                                               'reconstructed waterfall.')

    def test_callback(self):
        machine = mch.Machine(**MACHINE_ARGS)

        denergy = np.array([-115567.32591061])
        dphi = np.array([0.33178332])
        nparts = 1

        recprof = 5
        nturns = machine.dturns * (machine.nprofiles - 1)

        rfv1 = machine.vrf1_at_turn * machine.q
        rfv2 = machine.vrf2_at_turn * machine.q

        xp = np.zeros((machine.nprofiles, nparts))
        yp = np.zeros((machine.nprofiles, nparts))

        results = []

        def callback(progress: int, total: int):
            results.append(progress)

        libtomo.kick_and_drift(
            xp, yp, denergy, dphi, rfv1, rfv2, machine, recprof,
            nturns, nparts, callback=callback)

        correct = np.arange(1, nturns+1).tolist()

        self.assertListEqual(results, correct)

    def _load_waterfall(self):
        base_dir = os.path.split(os.path.realpath(__file__))[0]
        base_dir = os.path.split(base_dir)[0]
        data_path = os.path.join(base_dir, 'resources')

        waterfall = np.load(os.path.join(
            data_path, 'waterfall_INDIVShavingC325.npy'))
        return waterfall
