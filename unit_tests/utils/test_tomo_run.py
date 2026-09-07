"""Unit-tests for the tomo_run module.

Run as python test_tomo_run.py in console or via coverage
"""

import os
import unittest

import numpy as np
import numpy.testing as nptest

import longitudinal_tomography.utils.tomo_run as tomorun
from longitudinal_tomography.utils import tomo_config as conf

# Maximum relative deviation between reconstructed and reference values.
MAX_DEV_FACTOR = 1e-6

# Values retrieved from INDIVShavingC325.dat
NBINS = 254                 # 760 frame bins rebinned by 3
DTBIN = 3e-9                # 1e-9 s frame bins rebinned by 3
DEBIN = 3698.1544291395694
FILMSTART = 0               # reconstructed profile when none is given


class TestTomoRun(unittest.TestCase):
    """Full reconstruction from input file to phase space image."""

    @classmethod
    def setUpClass(cls):
        conf.AppConfig.use_cpu()
        conf.AppConfig.set_double_precision()

        base_dir = os.path.split(os.path.realpath(__file__))[0]
        base_dir = os.path.split(base_dir)[0]
        cls.input_path = os.path.join(base_dir, 'resources',
                                      'INDIVShavingC325.dat')

        cls.t_range, cls.e_range, cls.phase_space = tomorun.run(cls.input_path)

    def test_run_output_shapes(self):
        self.assertEqual(self.t_range.shape, (NBINS,),
                         msg='Time axis has the wrong length')
        self.assertEqual(self.e_range.shape, (NBINS,),
                         msg='Energy axis has the wrong length')
        self.assertEqual(self.phase_space.shape, (NBINS, NBINS),
                         msg='Phase space image has the wrong shape')

    def test_run_phase_space_normalized(self):
        self.assertAlmostEqual(
            self.phase_space.sum(), 1.0,
            msg='Reconstructed phase space was not normalized')
        self.assertGreaterEqual(
            self.phase_space.min(), 0.0,
            msg='Negative areas were not removed from the phase space')

    def test_run_time_axis(self):
        nptest.assert_allclose(
            np.diff(self.t_range), DTBIN, rtol=MAX_DEV_FACTOR,
            err_msg='Time axis is not spaced by the rebinned bin width')
        nptest.assert_allclose(
            [self.t_range[0], self.t_range[-1]], [-3.34e-07, 4.25e-07],
            rtol=MAX_DEV_FACTOR, err_msg='Error in the time axis limits')

    def test_run_energy_axis(self):
        nptest.assert_allclose(
            np.diff(self.e_range), DEBIN, rtol=MAX_DEV_FACTOR,
            err_msg='Energy axis is not spaced by dEbin')
        nptest.assert_allclose(
            [self.e_range[0], self.e_range[-1]],
            [-469665.6125007296, 465967.45807159005],
            rtol=MAX_DEV_FACTOR, err_msg='Error in the energy axis limits')

    def test_run_phase_space_values(self):
        self.assertEqual(
            np.unravel_index(np.argmax(self.phase_space),
                             self.phase_space.shape), (98, 114),
            msg='Phase space density peaks in the wrong bin')
        nptest.assert_allclose(
            self.phase_space.max(), 6.612865769143491e-05,
            rtol=MAX_DEV_FACTOR,
            err_msg='Error in the peak phase space density')
        nptest.assert_allclose(
            self.phase_space[100, 100], 5.22441557604237e-05,
            rtol=MAX_DEV_FACTOR,
            err_msg='Error in the reconstructed phase space density')

    def test_run_reconstruct_profile(self):
        recprof = FILMSTART + 5
        t_range, e_range, phase_space = tomorun.run(
            self.input_path, reconstruct_profile=recprof)

        nptest.assert_allclose(
            t_range, self.t_range, rtol=MAX_DEV_FACTOR,
            err_msg='Time axis should not depend on the reconstructed profile')
        nptest.assert_allclose(
            e_range, self.e_range, rtol=MAX_DEV_FACTOR,
            err_msg='Energy axis should not depend on the reconstructed '
                    'profile')
        self.assertAlmostEqual(
            phase_space.sum(), 1.0,
            msg='Reconstructed phase space was not normalized')
        self.assertFalse(
            np.array_equal(phase_space, self.phase_space),
            msg='Reconstructing another profile gave an identical phase space')


if __name__ == '__main__':
    unittest.main()
