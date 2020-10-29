"""Unit-tests for the Tracking class.

Run as python test_tracking.py in console or via coverage
"""

import os
import unittest

import numpy as np

import tomo.data.profiles as prof
import tomo.tracking.machine as mch
import tomo.tracking.tracking as tck
from tomo import exceptions as expt

# Machine arguments based on the input file INDIVShavingC325.dat
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


class TestTracker(unittest.TestCase):

    def test_parameter_notMachine_fails(self):
        with self.assertRaises(
                expt.MachineParameterError, msg='arguments not being of type '
                                                'Machine should raise '
                                                'an Exception'):
            tracker = tck.Tracking(None)

    def test_tracking_aut_distr(self):
        machine = mch.Machine(**MACHINE_ARGS)

        # Tracking only a few particles for 20 time frames.
        machine.snpt = 1
        rbn = 13
        machine.rbn = rbn
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.nprofiles = 20

        machine.values_at_turns()
        
        tracker = tck.Tracking(machine)
        xp, yp = tracker.track(recprof=10)

        # Comparing the coordinates of particle #0 only.
        correct_x = [-19.432295466713434, -19.3925512994595, 
                     -19.35163208683023, -19.309627946073313,
                     -19.266631745225464, -19.222738757376533,
                     -19.178046284007912, -19.132653251008257,
                     -19.086659781737467, -19.04016675218461,
                     -18.99327533382096, -18.946086530163132,
                     -18.898700713317176, -18.851217166861232,
                     -18.803733641339306, -18.75634592838512,
                     -18.709147459083823, -18.662228931627606,
                     -18.615677972651334, -18.5695788358727]

        correct_y = [-233614.99041969655, -240836.66283342137,
                     -247530.01310891184, -253677.7708608252,
                     -259264.64084817207, -264277.4952538157,
                     -268705.5465303423, -272540.4960483705,
                     -275776.6542680894, -278411.0288037383,
                     -280443.3775430866, -281876.2248907341,
                     -282714.8401894965, -282967.17839776503,
                     -282643.784119737, -281757.6610572577,
                     -280324.1098370269, -278360.53793064784,
                     -275886.24600012705, -272922.19544914237]

        for x, cx in zip(xp[:, 0], correct_x):
            self.assertAlmostEqual(
                x, cx, msg='Error in tracking of particle '
                           'found in x-coordinate')
        for y, cy in zip(yp[:, 0], correct_y):
                self.assertAlmostEqual(
                    y, cy, msg='Error in tracking of particle '
                               'found in y-coordinate')

    def test_tracking_man_distr(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 20
        machine.values_at_turns()
        
        phase_0 = np.array([0.33178332])
        energy_0 = np.array([-115567.32591061])
        in_coordinates = (phase_0, energy_0)

        tracker = tck.Tracking(machine)
        xp, yp = tracker.track(recprof=10, init_distr=in_coordinates)

        correct_x = [0.11217504, 0.13585361, 0.1592622, 0.1823566,
                     0.20509384, 0.22743227, 0.24933161, 0.27075299,
                     0.29165905, 0.31201393, 0.33178332, 0.35093449,
                     0.36943629, 0.38725915, 0.40437511, 0.4207578,
                     0.4363824, 0.45122569, 0.46526597, 0.47848306]
        
        for x, cx in zip(xp[:, 0], correct_x):
            self.assertAlmostEqual(
                x, cx, msg='Error in tracking of particle '
                           'found in x-coordinate')

        correct_y = [-141501.80292005, -140012.42084442, -138256.87228931,
                     -136242.257239, -133976.14426166, -131466.51658527,
                     -128721.71792799, -125750.39872057, -122561.4633275,
                     -119164.01883305, -115567.32591061, -111780.75223935,
                     -107813.72887332, -103675.70990658, -99376.1357149,
                     -94924.39999134, -90329.82073156, -85601.61526492,
                     -80748.87937168,  -75780.57047406]
        for y, cy in zip(yp[:, 0], correct_y):
            self.assertAlmostEqual(
                y, cy, msg='Error in tracking of particle '
                           'found in y-coordinate')


    def test_self_field_tracking(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.zwall_over_n = 50.0
        machine.g_coupling = 1.0
        machine.values_at_turns()

        base_dir = os.path.split(os.path.realpath(__file__))[0]
        base_dir = os.path.split(base_dir)[0]
        data_path = os.path.join(base_dir, 'resources')

        waterfall = np.load(os.path.join(
                        data_path, 'waterfall_INDIVShavingC325.npy'))
        vself = np.load(os.path.join(
                        data_path, 'vself_INDIVShavingC325.npy'))

        profiles = prof.Profiles(machine, machine.dtbin, waterfall)

        profiles.phiwrap = 6.283185307179586
        profiles.vself = vself

        rbn = 3
        machine.dtbin *= rbn
        machine.nbins = waterfall.shape[1]
        machine.synch_part_x /= rbn

        phase_0 = np.array([0.33178332])
        energy_0 = np.array([-115567.32591061])

        tracker = tck.Tracking(machine)
        tracker.enable_self_fields(profiles)

        tracker.particles.xorigin = -82.20164208806912
        tracker.particles.dEbin = 3698.1544291396035
        xp, yp = tracker.track(50, (phase_0, energy_0))

        xp_0 = 73.58628603566842
        xp_50 = 135.37116548104402
        xp_149 = 73.43430438854142

        yp_0 = 130.08894709334083
        yp_50 = 95.75000000000071
        yp_149 = 120.6218531014772

        correct_y = [yp_0, yp_50, yp_149]
        test_y = [float(yp[0]), float(yp[50]), float(yp[149])]
        for y, cy in zip(test_y, correct_y):
            self.assertAlmostEqual(
                y, cy, msg='An error was found in the y-coordinates '
                           'tracked using self-fields.')

        correct_x =[xp_0, xp_50, xp_149]
        test_x = [float(xp[0]), float(xp[50]), float(xp[149])]
        for x, cx in zip(test_x, correct_x):
            self.assertAlmostEqual(
                x, cx, msg='An error was found in the x-coordinates '
                           'tracked using self-fields.')

    def test_self_field_flag_fails(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        # Dummy profiles object, waterfall is array of ones.
        profiles = prof.Profiles(machine, machine.dtbin, np.ones((150, 1)))

        tracker = tck.Tracking(machine)

        with self.assertRaises(expt.SelfFieldTrackingError,
                               msg='Missing self field voltages '
                                   'should raise an error'):
            tracker.enable_self_fields(profiles)

    def test_kick_and_drift_hybrid_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.nprofiles = 10
        machine.values_at_turns()
        
        phase_0 = np.array([0.33178332])
        energy_0 = np.array([-115567.32591061])
        
        rfv1 = machine.vrf1_at_turn * machine.q
        rfv2 = machine.vrf2_at_turn * machine.q
        
        tracker = tck.Tracking(machine)
        xp, yp = tracker.kick_and_drift(
                    energy_0, phase_0, rfv1, rfv2, rec_prof=5)
        
        correct_x = [0.22739336232235102, 0.24930078456175753,
                     0.27073013058527245, 0.29164399861825624,
                     0.3120065061037849, 0.33178332,
                     0.350941678970489, 0.3694504079276061,
                     0.3872799254524489, 0.40440224466848473]
        
        correct_y = [-131465.5660292741, -128721.12723284948,
                     -125750.0786811894, -122561.328959532,
                     -119163.98960288391, -115567.32591061,
                     -111780.71031054154, -107813.57867853255,
                     -103675.38995760499, -99375.58935765548]

        for x, cx in zip(xp, correct_x):
            self.assertAlmostEqual(
                float(x), cx, msg='An error was found in the x-coordinates '
                                  'tracked using self-fields.')
        for y, cy in zip(yp, correct_y):
            self.assertAlmostEqual(
                float(y), cy, msg='An error was found in the y-coordinates '
                                  'tracked using self-fields.')
