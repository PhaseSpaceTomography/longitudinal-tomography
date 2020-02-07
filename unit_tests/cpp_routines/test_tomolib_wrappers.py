'''Unit-tests for the funtions in the tomolib_wrappers module.

Run as python test_tomolib_wrappers.py in console or via coverage
'''

import numpy as np
import numpy.testing as nptest
import os
import unittest

import tomo.cpp_routines.tomolib_wrappers as tlw
import tomo.utils.exceptions as expt
import tomo.tracking.machine as mch


# Machine arguments mased on the input file INDIVShavingC325.dat
MACHINE_ARGS = {
    'output_dir':          '/tmp/',
    'dtbin':               9.999999999999999E-10,
    'dturns':              5,
    'synch_part_x':        334.00000000000006,
    'demax':               -1.E6,
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
    'max_dt':              9.999999999999999E-10 * 760 # dtbin * nbins
    }


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


        new_denergy = tlw.kick(machine, denergy, dphi,
                               rfv1, rfv2, npart, turn, up=True)

        correct_energy = -113495.65825924404
        self.assertAlmostEqual(
            new_denergy[0], correct_energy,
            msg='Kick in upward direction was calculated incorectly')

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


        new_denergy = tlw.kick(machine, denergy, dphi,
                               rfv1, rfv2, npart, turn, up=False)

        correct_energy = -116610.12118255378
        self.assertAlmostEqual(
            new_denergy[0], correct_energy,
            msg='Kick in downward direction was calculated incorectly')

    def test_drift_up_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 10
        machine.values_at_turns()

        denergy = np.array([-115567.32591061])
        dphi = np.array([0.33178332])
        nparts = 1
        turn = 0

        new_dphi = tlw.drift(denergy, dphi, machine.drift_coef,
                             nparts, turn, up=True)

        correct_dphi = 0.3356669466375665
        self.assertAlmostEqual(
            new_dphi[0], correct_dphi,
            msg='Drift in upward direction was calculated incorectly')

    def test_drift_down_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 10
        machine.values_at_turns()

        denergy = np.array([-115567.32591061])
        dphi = np.array([0.33178332])
        nparts = 1
        turn = (machine.nprofiles - 1) * machine.dturns

        new_dphi = tlw.drift(denergy, dphi, machine.drift_coef,
                             nparts, turn, up=False)

        correct_dphi = 0.3279023169434031
        self.assertAlmostEqual(
            new_dphi[0], correct_dphi,
            msg='Drift in downward direction was calculated incorectly')

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

        xp, yp = tlw.kick_and_drift(
                        xp, yp, denergy, dphi, rfv1, rfv2, recprof,
                        nturns, nparts, machine=machine)
        
        correct_xp = np.array([[0.22739336], [0.24930078], [0.27073013],
                               [0.291644], [0.31200651], [0.33178332],
                               [0.35094168], [0.36945041], [0.38727993],
                               [0.40440224]])
        correct_yp = np.array([[-131465.56602927], [-128721.12723285],
                               [-125750.07868119], [-122561.32895953],
                               [-119163.98960288], [-115567.32591061],
                               [-111780.71031054], [-107813.57867853],
                               [-103675.3899576], [ -99375.58935766]])
        
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

        xp, yp = tlw.kick_and_drift(
                        xp, yp, denergy, dphi, rfv1, rfv2, recprof,
                        nturns, nparts, phi0, deltaE0, omega_rev0,
                        drift_coef, phi12, h_ratio, dturns)
        
        correct_xp = np.array([[0.22739336], [0.24930078], [0.27073013],
                               [0.291644], [0.31200651], [0.33178332],
                               [0.35094168], [0.36945041], [0.38727993],
                               [0.40440224]])
        correct_yp = np.array([[-131465.56602927], [-128721.12723285],
                               [-125750.07868119], [-122561.32895953],
                               [-119163.98960288], [-115567.32591061],
                               [-111780.71031054], [-107813.57867853],
                               [-103675.3899576], [ -99375.58935766]])
        
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

        with self.assertRaises(expt.InputError,
                               msg='Too few arrays should '
                                   'raise an exception'):
            xp, yp = tlw.kick_and_drift(
                            xp, yp, denergy, dphi, rfv1, rfv2, recprof,
                            nturns, nparts, phi0, deltaE0, omega_rev0,
                            drift_coef, phi12, h_ratio)

        dturns = machine.dturns
        some_useless_var = None
        with self.assertRaises(expt.InputError,
                               msg='Too many arrays should '
                                   'raise an exception'):
            xp, yp = tlw.kick_and_drift(
                            xp, yp, denergy, dphi, rfv1, rfv2, recprof,
                            nturns, nparts, phi0, deltaE0, omega_rev0,
                            drift_coef, phi12, h_ratio)        

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
        
        tlw.back_project(weights, flat_points, flat_profs, nparts, nprofs)

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

        rec = tlw.project(np.zeros(waterfall_shape), flat_points,
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

    def test_reconstruct_correct(self):
        nprofs = 10
        nparts = 50
        nbins = 100
        niter = 1

        weights = np.zeros(nparts)
        discr = np.zeros(niter+1)

        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]
        xp = xp.T
        flat_points = xp.copy()
        for i in range(nprofs):
            flat_points[:, i] += nbins * i
        flat_points = np.ascontiguousarray(flat_points).astype(np.int32)

        waterfall = self._load_waterfall()
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        flat_profs = np.ascontiguousarray(
                        waterfall.flatten()).astype(np.float64)

        weights, discr = tlw.reconstruct(
                            weights, xp, flat_profs, discr, niter,
                            nbins, nparts, nprofs, verbose=False)

        correct_w = np.array([11.91929074, 12.28856519, 12.28856519,
                              12.46441016, 12.72817762, 12.72817762,
                              12.81610011, 12.56991715, 12.53474815,
                              12.41165667, 12.35890318, 12.28856519,
                              12.27098069, 12.1830582,  12.2182272,
                              12.34131868, 12.28856519, 12.2182272,
                              12.51716365, 12.60508614, 12.71059313,
                              12.93919159, 13.20295905, 13.41397302,
                              13.53706451, 13.80083197, 14.11735292,
                              14.32836689, 14.62730335, 14.57454985,
                              14.69764134, 14.68005684, 14.48662737,
                              14.48662737, 14.34595139, 14.32836689,
                              14.32836689, 14.17010641, 14.09976842,
                              13.99426144, 13.90633895, 13.80083197,
                              13.589818,   13.6074025,  13.589818,  
                              13.32605054, 12.93919159, 12.60508614,
                              12.25339619, 11.93687524])
        
        correct_discr = np.array([0.10268103, 0.10267469])

        nptest.assert_almost_equal(
            weights, correct_w,
            err_msg='Error in weights after reconstruction.')

        nptest.assert_almost_equal(
            discr, correct_discr,
            err_msg='Error in calculated discrepancies after reconstruction.')

    def _load_waterfall(self):
        base_dir = os.path.split(os.path.realpath(__file__))[0]
        base_dir = os.path.split(base_dir)[0]
        data_path = os.path.join(base_dir, 'resources')
    
        waterfall = np.load(os.path.join(
                        data_path, 'waterfall_INDIVShavingC325.npy'))
        return waterfall
