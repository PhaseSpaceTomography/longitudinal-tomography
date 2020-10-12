"""Unit-tests for the tomo_output module.

Run as python test_tomo_output.py in console or via coverage
"""

import os
import shutil
import unittest

import numpy as np
import numpy.testing as nptest

import tomo.tracking.machine as mch
import tomo.tracking.particles as pts
import tomo.utils.exceptions as expt
import tomo.utils.tomo_output as tout

base_dir = os.path.split(os.path.realpath(__file__))[0]
base_dir = os.path.split(base_dir)[0]
tmp_dir = os.path.join(base_dir, 'tmp')

# All values retrieved from INDIVShavingC325.dat
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


class TestTomoOut(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    def test_save_profile_ftn(self):
        waterfall = np.arange(25).reshape((5, 5))

        recprof = 2
        tout.save_profile_ftn(waterfall, 2, tmp_dir)

        profile_pth = os.path.join(tmp_dir, f'profile{recprof + 1:03d}.data')
        with open(profile_pth, 'r') as f:
            read = f.readlines()

        correct = [' 1.0000000E+01\n', ' 1.1000000E+01\n', ' 1.2000000E+01\n',
                   ' 1.3000000E+01\n', ' 1.4000000E+01\n']

        nptest.assert_equal(
            read, correct, err_msg='Format of Fortran styled profiles file '
                                   'is incorrect')

    def test_save_vself_ftn(self):
        self_volts = np.arange(9).reshape((3, 3))

        tout.save_self_volt_profile_ftn(self_volts, tmp_dir)

        profile_pth = os.path.join(tmp_dir, 'vself.data')
        with open(profile_pth, 'r') as f:
            read = f.readlines()

        correct = [' 0.0000000E+00\n', ' 1.0000000E+00\n', ' 2.0000000E+00\n',
                   ' 3.0000000E+00\n', ' 4.0000000E+00\n', ' 5.0000000E+00\n',
                   ' 6.0000000E+00\n', ' 7.0000000E+00\n', ' 8.0000000E+00\n']

        nptest.assert_equal(
            read, correct, err_msg='Format of Fortran styled vself file '
                                   'is incorrect')

    def test_save_phase_space_ftn(self):
        image = np.arange(9).reshape((3, 3))
        recprof = 2

        tout.save_phase_space_ftn(image, recprof, tmp_dir)

        image_pth = os.path.join(tmp_dir, f'image{recprof + 1:03d}.data')
        with open(image_pth, 'r') as f:
            read = f.readlines()

        correct = ['  0.0000000E+00\n', '  1.0000000E+00\n',
                   '  2.0000000E+00\n', '  3.0000000E+00\n',
                   '  4.0000000E+00\n', '  5.0000000E+00\n',
                   '  6.0000000E+00\n', '  7.0000000E+00\n',
                   '  8.0000000E+00\n']

        nptest.assert_equal(
            read, correct, err_msg='Format of Fortran styled image file '
                                   'is incorrect')

    def test_save_difference_ftn(self):
        diff = np.arange(11) * np.pi / 100
        recprof = 2
        tout.save_difference_ftn(diff, tmp_dir, recprof)

        diff_pth = os.path.join(tmp_dir, f'd{recprof + 1:03d}.data')
        with open(diff_pth, 'r') as f:
            read = f.readlines()

        correct = ['           0  0.0000000E+00\n',
                   '           1  3.1415927E-02\n',
                   '           2  6.2831853E-02\n',
                   '           3  9.4247780E-02\n',
                   '           4  1.2566371E-01\n',
                   '           5  1.5707963E-01\n',
                   '           6  1.8849556E-01\n',
                   '           7  2.1991149E-01\n',
                   '           8  2.5132741E-01\n',
                   '           9  2.8274334E-01\n',
                   '          10  3.1415927E-01\n']

        nptest.assert_equal(read, correct,
                            err_msg='Discrepancies was saved incorrectly.')

    def test_write_plotinfo_ftn(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()
        particles = pts.Particles()
        particles.homogeneous_distribution(machine, 0)
        profile_charge = 1e6
        
        plotinfo = tout.write_plotinfo_ftn(machine, particles, profile_charge)
        plotinfo_list = np.array(plotinfo.split('\n'))
        
        correct = [' plotinfo.data',
                   'Number of profiles used in each reconstruction,',
                   ' profilecount = 150',
                   'Width (in pixels) of each image = length (in bins) '
                   'of each profile,',
                   ' profilelength = 760',
                   'Width (in s) of each pixel = width of each profile bin,',
                   ' dtbin = 1.0000E-09',
                   'Height (in eV) of each pixel,',
                   ' dEbin = 1.2327E+03',
                   'Number of elementary charges in each image,',
                   ' eperimage = 1.000E+06',
                   'Position (in pixels) of the reference synchronous point:',
                   ' xat0 =  334.000',
                   ' yat0 =  380.000',
                   'Foot tangent fit results (in bins):',
                   ' tangentfootl =    0.000',
                   ' tangentfootu =    0.000',
                   ' fit xat0 =   0.000',
                   'Synchronous phase (in radians):',
                   ' phi0( 1) = 0.4008',
                   'Horizontal range (in pixels) of the region in '
                   'phase space of map elements:',
                   ' imin( 1) =   0 and imax( 1) =  759']
        
        nptest.assert_equal(plotinfo_list, correct,
                            err_msg='Error in creation of plotinfo string')

    def test_write_plotinfo_ftn_fitted_x(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        fittedx = 23.123445677
        tanbin_low = 3.12345567
        tanbin_up = 80.12345567  
        machine.load_fitted_synch_part_x_ftn((fittedx, tanbin_low, tanbin_up))

        particles = pts.Particles()
        particles.homogeneous_distribution(machine, 0)
        profile_charge = 1e6
        
        plotinfo = tout.write_plotinfo_ftn(machine, particles, profile_charge)
        plotinfo_list = np.array(plotinfo.split('\n'))
        
        correct = [' plotinfo.data',
                   'Number of profiles used in each reconstruction,',
                   ' profilecount = 150',
                   'Width (in pixels) of each image = length (in bins) '
                   'of each profile,',
                   ' profilelength = 760',
                   'Width (in s) of each pixel = width of each profile bin,',
                   ' dtbin = 1.0000E-09',
                   'Height (in eV) of each pixel,',
                   ' dEbin = 1.2327E+03',
                   'Number of elementary charges in each image,',
                   ' eperimage = 1.000E+06',
                   'Position (in pixels) of the reference synchronous point:',
                   ' xat0 =  23.123',
                   ' yat0 =  380.000',
                   'Foot tangent fit results (in bins):',
                   ' tangentfootl =    3.123',
                   ' tangentfootu =    80.123',
                   ' fit xat0 =   23.123',
                   'Synchronous phase (in radians):',
                   ' phi0( 1) = 0.4008',
                   'Horizontal range (in pixels) of the region in '
                   'phase space of map elements:',
                   ' imin( 1) =   0 and imax( 1) =  759']
        
        print(plotinfo_list)

        nptest.assert_equal(plotinfo_list, correct,
                            err_msg='Error in creation of plotinfo string. '
                                    'fitted value for synch. part. x '
                                    'was provided.')

    def test_write_plotinfo_ftn_no_dEbin_fails(self):
        machine = mch.Machine(**MACHINE_ARGS)
        particles = pts.Particles()
        particles.imin = 0
        particles.imax = 100
        profile_charge = 1e6

        with self.assertRaises(expt.EnergyBinningError,
                               msg='An exception should be raised if the '
                                   'particles.dEbin is None'):
            tout.write_plotinfo_ftn(machine, particles, profile_charge)

    def test_write_plotinfo_ftn_no_imin_imax_fails(self):
        machine = mch.Machine(**MACHINE_ARGS)
        particles = pts.Particles()
        particles.imin = None
        particles.imax = 100
        particles.dEbin = 1e3
        profile_charge = 1e6

        with self.assertRaises(expt.PhaseLimitsError,
                               msg='An exception should be raised if '
                                   'particles.imin is None'):
            tout.write_plotinfo_ftn(machine, particles, profile_charge)

        particles.imin = 0
        particles.imax = None

        with self.assertRaises(expt.PhaseLimitsError,
                               msg='An exception should be raised if '
                                   'particles.imax is None'):
            tout.write_plotinfo_ftn(machine, particles, profile_charge)

    def test_create_phase_space_image(self):

        weights = np.ones(25)

        ax = np.arange(5)
        xp, yp = np.meshgrid(ax, ax)
        xp = np.vstack(xp.flatten())
        yp = np.vstack(yp.flatten())

        nbins = 5
        recprof = 0
        img = tout.create_phase_space_image(xp, yp, weights, nbins, recprof)

        correct = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04]])

        nptest.assert_equal(
            img, correct, err_msg='Phase space image was created incorrectly')
