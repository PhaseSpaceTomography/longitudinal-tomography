import unittest
import numpy as np
import numpy.testing as nptest
import os
from tomo.time_space import TimeSpace
from tomo.map_info import MapInfo
from tomo.parameters import Parameters
from tomo.tracking.tracking import Tracking
from tomo.cpp_routines.tomolib_wrappers import kick, drift
from unit_tests.C500values import C500

resources_dir = os.path.realpath(__file__)
resources_dir = '/'.join(resources_dir.split('/')[:-1])
resources_dir += '/resources/particle_tracking' 


class TestTrack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c500 = C500()

        input_file = cls.c500.path + 'C500MidPhaseNoise.dat'
        with open(input_file, 'r') as f:
            read = f.readlines()
        
        cls.raw_param = read[:98]
        cls.raw_data = np.array(read[98:], dtype=float)
        cls.par = Parameters()
        cls.par.parse_from_txt(cls.raw_param)
        cls.par.fill()


    # def test_init_tracked_point(self):
    #     rv = TestTrack.rec_vals
    #     cv = TestTrack.c500.values
    #     ca = TestTrack.c500.arrays

    #     xp = np.zeros(int(np.ceil(rv['needed_maps'] * cv['snpt']**2
    #                               / cv['profile_count'])))
    #     yp = np.zeros(int(np.ceil(rv['needed_maps'] * cv['snpt']**2
    #                               / cv['profile_count'])))

    #     xp, yp = Tracking._init_tracked_point(
    #                         cv['snpt'], ca['imin'][0],
    #                         ca['imax'][0], ca['jmin'],
    #                         ca['jmax'], xp, yp,
    #                         rv['points'][0], rv['points'][1])

    #     nptest.assert_equal(xp, rv['init_xp'],
    #                         err_msg='Error in initiating'\
    #                                 ' of tracked points (xp)')
    #     nptest.assert_almost_equal(yp, rv['init_yp'],
    #                                err_msg='Error in initiating of'\
    #                                        ' tracked points (yp)')
    #     self.assertEqual(len(xp), 406272,
    #                      msg='Error in number of used pixels')

    # def test_populate_bins(self):
    #     rv = TestTrack.rec_vals
    #     cv = TestTrack.c500.values
    #     points = Tracking._populate_bins(Tracking, cv["snpt"])
    #     nptest.assert_equal(points, rv["points"],
    #                         err_msg="Initial points calculated incorrectly.")

    # def test_calc_dphi_denergy(self):
    #     cv = TestTrack.c500.values
    #     ca = TestTrack.c500.arrays
    #     rv = TestTrack.rec_vals

    #     # Creating timespace object with needed values
    #     ts = TimeSpace.__new__(TimeSpace)
    #     ts.par = Parameters()
    #     ts.x_origin = cv['xorigin']
    #     ts.par.h_num = cv['h_num']
    #     ts.par.omega_rev0 = ca['omegarev0']
    #     ts.par.dtbin = cv['dtbin']
    #     ts.par.phi0 = ca['phi0']
    #     ts.par.yat0 = cv['yat0']

    #     # Creating mapinfo object with needed values        
    #     track = Tracking(ts, self.mi)

    #     dphi, denergy = track.coords_to_physical(np.array([100.125, 101.250]),
    #                                              np.array([19.125, 19.125]))
    #     correct_dphi = [0.20686511, 0.22605878]
    #     correct_denergy = [-1946025.30046682, -1946025.30046682]

    #     nptest.assert_almost_equal(dphi, correct_dphi,
    #                                err_msg='dphi was not '\
    #                                        'calculated correctly.')
    #     nptest.assert_almost_equal(denergy, correct_denergy,
    #                                err_msg='denergy was not '\
    #                                        'calculated correctly.')

    # def test_find_nr_pts(self):
    #     # Filling time space object with needed values for calc.
    #     ts = TimeSpace.__new__(TimeSpace)
    #     ts.par = Parameters()
    #     ts.par.snpt = self.c500.values['snpt']

    #     track = Tracking(ts, self.mi)

    #     npts = track.find_nr_of_particles()
    #     correct_npts = 406272

    #     self.assertEqual(npts, correct_npts,
    #                      msg='Error in calculation of needed particles')

    def test_kick(self):
        cv = self.c500.values
        ca = self.c500.arrays
        pr = Parameters()
        pr.x_origin = cv['xorigin']
        pr.h_num = cv['h_num']
        pr.omega_rev0 = ca['omegarev0']
        pr.dtbin = cv['dtbin']
        pr.phi0 = ca['phi0']
        pr.yat0 = cv['yat0']
        pr.deltaE0 = ca['deltaE0']

        # Values for kicking one test particle for one machine turn
        # Arbitrary numbers for dphi, denergy
        denergy = np.array([-2366156.6996680484])
        dphi = np.array([-1.48967785202453])
        rfv1 = np.array([0, 7945.403672852664])
        rfv2 = np.array([0, 0.0])
        turn = 1
        nparts = 1

        denergy = kick(pr, denergy, dphi, rfv1, rfv2, nparts, turn)
        
        correct_denergy = -2375933.37806776
        self.assertAlmostEqual(denergy[0], correct_denergy,
                               msg='denergy was calculated incorrectly '\
                                   'using "kick" function (cpp).')

    def test_drift(self):
        
        # Values for tracking drift for one particle one machine turn turn
        # Arbitrary numbers for dphi, denergy
        dphase = self.c500.arrays['dphase']
        denergy = np.array([-2366156.6996680484])
        dphi = np.array([-1.4992388704498583])
        rfv1 = np.array([7945.403672852664])
        rfv2 = np.array([0.0])
        turn = 0
        nparts = 1

        dphi = drift(denergy, dphi, dphase, nparts, turn)

        correct_dphi = -1.48967785202453
        self.assertAlmostEqual(dphi[0], correct_dphi,
                               msg='dphi was calculated incorrectly '\
                                   'using "drift" function (cpp).')

    def test_kick_and_drift(self):
        rfv1 = (self.par.vrf1 + self.par.vrf1dot
                * self.par.time_at_turn * self.par.q)
        rfv2 = np.zeros(rfv1.shape)

        dphi = np.array([0.20473248, 0.20473248, 0.20473248,
                         -0.22179352, 0.20473248, 0.63125847])
        denergy = np.array([-641867.41544633, -58351.58322239, 525164.24900154,
                            -58351.58322239, -58351.58322239, -58351.58322239])

        tracker = Tracking(self.par)
        alldphi, alldenergy = tracker.kick_and_drift(denergy, dphi, rfv1, rfv2)

        correct_dphi = np.load(f'{resources_dir}/few_xp_tracked_phys.npy')
        correct_denergy = np.load(f'{resources_dir}/few_yp_tracked_phys.npy')

        nptest.assert_almost_equal(alldphi, correct_dphi, 1,
                                   err_msg='Error in kick and drift routine')
        nptest.assert_almost_equal(alldenergy, correct_denergy, 1,
                                   err_msg='Error in kick and drift routine')


    # Teesting by tracking one particle (particle #0 in C500MidPhaseNoise)
    def old_test_kick_and_drift_self_field(self):
        

        cv = TestTrack.c500.values
        ca = TestTrack.c500.arrays
        ts = TimeSpace.__new__(TimeSpace)
        ts.par = Parameters()
        ts.x_origin = cv['xorigin']
        ts.par.h_num = cv['h_num']
        ts.par.omega_rev0 = ca['omegarev0']
        ts.par.dtbin = cv['dtbin']
        ts.par.phi0 = ca['phi0']
        ts.par.yat0 = cv['yat0']
        ts.par.deltaE0 = ca['deltaE0']
        ts.par.dphase = ca['dphase']
        ts.par.dturns = cv['dturns'] 
        ts.par.phiwrap = cv['phiwrap']
        ts.par.q = cv['q']
        ts.vself = ca['vself']


        nturns = cv['dturns'] * (cv['profile_count'] - 1)
        nparts = 1
        
        denergy = np.array([-218818.4370839749])
        dphi = np.array([-1.4651167908948117])

        rfv1 = (cv['vrf1'] + cv['vrf1dot'] * ca['time_at_turn']) * cv['q']
        rfv2 = np.zeros(nturns + 1)

        xp = np.zeros(cv['profile_count'])
        yp = np.zeros(cv['profile_count'])

        track = Tracking(ts, self.mi)
        track.kick_and_drift_self(xp, yp, denergy, dphi, rfv1,
                                  rfv2, nturns, nparts)

        correct_xp_last_turn = 110.5905464803269
        correct_yp_last_turn = 184.19385407895095

        self.assertAlmostEqual(xp[-1], correct_xp_last_turn,
                               msg='The tracking of a particle went wrong!\n'\
                                   'Particle ended up at the '\
                                   'wrong x coordinate.')

        self.assertAlmostEqual(yp[-1], correct_yp_last_turn,
                               msg='The tracking of a particle went wrong!\n'\
                                   'Particle ended up at the '\
                                   'wrong y coordinate.')

    def test_calc_xp_self_field(self):
        cv = TestTrack.c500.values
        ca = TestTrack.c500.arrays
        dphi = -1.4642326030009774
        turn = 1

        xp = Tracking._calc_xp_sf(dphi, ca['phi0'][turn], cv['xorigin'],
                                 cv['h_num'], ca['omegarev0'][turn],
                                 cv['dtbin'], cv['phiwrap'])

        correct_xp = 2.176945309903646

        self.assertAlmostEqual(xp, correct_xp,
                               msg='Error in calculation of xp in '\
                                    'while calculating with self fields')
