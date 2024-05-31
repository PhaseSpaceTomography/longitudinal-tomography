"""Unit-tests for the TomographyCpp class.

Run as python test_tomography_cpp.py in console or via coverage
"""
from __future__ import annotations
import os
import unittest

import numpy as np
import numpy.testing as nptest

import longitudinal_tomography.tomography.tomography as tmo
from longitudinal_tomography import exceptions as expt
from longitudinal_tomography.utils import tomo_config as conf

from .. import commons

SINGLE_PREC_ACCURACY_PLACES = 5


class TestTomographyCpp(unittest.TestCase):

    def setUp(self):
        conf.AppConfig.use_cpu()
        conf.AppConfig.set_double_precision()

    def test_set_xp_not_iterable_fails(self):
        waterfall = np.ones((10, 10))
        tomo = tmo.TomographyCpp(waterfall)

        with self.assertRaises(expt.CoordinateImportError,
                               msg='Providing x coordinates as non iterable '
                                   'should raise an Exception'):
            tomo.xp = 1

    def test_set_xp_one_dim_fails(self):
        waterfall = np.ones((10, 10))
        tomo = tmo.TomographyCpp(waterfall)

        with self.assertRaises(expt.CoordinateImportError,
                               msg='Providing x coordinates array having '
                                   'another number of dimensions than two '
                                   'should raise an Exception'):
            tomo.xp = [1, 2, 4]

    def test_set_xp_wrong_nprofs_fails(self):
        nprofs = 5
        nbins = 10
        waterfall = np.ones((nprofs, nbins))
        tomo = tmo.TomographyCpp(waterfall)

        nparts = 20
        pts_nprofs = 2
        with self.assertRaises(expt.CoordinateImportError,
                               msg='Providing x coordinates array which has '
                                   'been tracked trough another number of '
                                   'profiles than found in the waterfall '
                                   'should raise an Exception'):
            tomo.xp = np.ones((nparts, pts_nprofs))

    def test_set_xp_outside_of_img_upper_fails(self):
        nprofs = 5
        img_width = 10
        waterfall = np.ones((nprofs, img_width))
        tomo = tmo.TomographyCpp(waterfall)

        nparts = 20
        xp = np.ones((nparts, nprofs))
        xp[1, 2] = img_width
        with self.assertRaises(expt.XPOutOfImageWidthError,
                               msg='Providing x coordinates array outside '
                                   'of image width should raise an Exception'):
            tomo.xp = xp

    def test_set_xp_outside_of_img_lower_fails(self):
        nprofs = 5
        img_width = 10
        waterfall = np.ones((nprofs, img_width))
        tomo = tmo.TomographyCpp(waterfall)

        nparts = 20
        xp = np.ones((nparts, nprofs))
        xp[1, 2] = -1
        with self.assertRaises(expt.XPOutOfImageWidthError,
                               msg='Providing x coordinates array outside '
                                   'of image width should raise an Exception'):
            tomo.xp = xp

    def test_xp_can_be_set_to_none(self):
        nprofs = 5
        img_width = 10
        nparts = 20
        waterfall = np.ones((nprofs, img_width))
        xp = np.ones((nparts, nprofs))
        tomo = tmo.TomographyCpp(waterfall, xp)
        tomo.xp = None

        self.assertEqual(
            tomo.xp, None, msg='xp should be able to be set to None')

    def test_waterfall_reduced_to_zeros_fail(self):
        nprofs = 5
        img_width = 10
        waterfall = -np.ones((nprofs, img_width))

        with self.assertRaises(expt.WaterfallReducedToZero,
                               msg='Providing an all negative or '
                                   'zero waterfall should raise an Exception'):
            tomo = tmo.TomographyCpp(waterfall)

    def test_run_xp_is_none_fails(self):
        nprofs = 5
        img_width = 10
        waterfall = np.ones((nprofs, img_width))
        tomo = tmo.TomographyCpp(waterfall)

        with self.assertRaises(expt.CoordinateError,
                               msg='Calling the run function, with '
                                   'longitudinal_tomography.xp=None '
                                   'should raise an Exception'):
            tomo.run()

    def test_run_correct_weights(self):
        waterfall = commons.load_waterfall()

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]

        tomo = tmo.TomographyCpp(waterfall)
        tomo.xp = xp.T
        weights = tomo.run(1)

        cweights = np.array([0.00861438, 0.00880599, 0.00893367, 0.00909539,
                             0.00918903, 0.00922729, 0.00928258, 0.00928246,
                             0.00927822, 0.00929515, 0.00929504, 0.00921845,
                             0.0092014,  0.00918014, 0.00915893, 0.00914618,
                             0.00919728, 0.00912925, 0.00918038, 0.00924009,
                             0.00920603, 0.00930828, 0.00940621, 0.00953821,
                             0.00959356, 0.00969566, 0.00981058, 0.00988708,
                             0.00998924, 0.01007859, 0.01014255, 0.01023618,
                             0.01021914, 0.01028299, 0.01026588, 0.01021909,
                             0.01016806, 0.01015526, 0.01011276, 0.01008299,
                             0.00999782, 0.00992973, 0.00991267, 0.00985308,
                             0.00977646, 0.0097126,  0.00971697, 0.00967434,
                             0.00961899, 0.00961908])

        nptest.assert_almost_equal(
            weights, cweights, err_msg='Weights were calculated incorrectly')

    def test_run_correct_weights_singleprec(self):
        conf.AppConfig.set_single_precision()
        waterfall = commons.load_waterfall().astype(np.float32)

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]

        tomo = tmo.Tomography(waterfall)
        tomo.xp = xp.T
        weights = tomo.run(1)

        cweights = np.array([0.00861438, 0.00880599, 0.00893367, 0.00909539,
                             0.00918903, 0.00922729, 0.00928258, 0.00928246,
                             0.00927822, 0.00929515, 0.00929504, 0.00921845,
                             0.0092014,  0.00918014, 0.00915893, 0.00914618,
                             0.00919728, 0.00912925, 0.00918038, 0.00924009,
                             0.00920603, 0.00930828, 0.00940621, 0.00953821,
                             0.00959356, 0.00969566, 0.00981058, 0.00988708,
                             0.00998924, 0.01007859, 0.01014255, 0.01023618,
                             0.01021914, 0.01028299, 0.01026588, 0.01021909,
                             0.01016806, 0.01015526, 0.01011276, 0.01008299,
                             0.00999782, 0.00992973, 0.00991267, 0.00985308,
                             0.00977646, 0.0097126,  0.00971697, 0.00967434,
                             0.00961899, 0.00961908])

        nptest.assert_almost_equal(
            weights, cweights, decimal=SINGLE_PREC_ACCURACY_PLACES, err_msg='Weights were calculated incorrectly')

    def test_run_correct_diff(self):
        waterfall = commons.load_waterfall()

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]

        tomo = tmo.TomographyCpp(waterfall)
        tomo.xp = xp.T
        weights = tomo.run(1)

        correct = 0.009561478717303546
        self.assertAlmostEqual(tomo.diff[0], correct,
                               msg='Discrepancy calculated incorrectly')

    def test_run_correct_diff_singleprec(self):
        conf.AppConfig.set_single_precision()
        waterfall = commons.load_waterfall().astype(np.float32)

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]

        tomo = tmo.TomographyCpp(waterfall)
        tomo.xp = xp.T
        weights = tomo.run(1)

        correct = 0.009561478717303546
        self.assertAlmostEqual(tomo.diff[0], correct, places=SINGLE_PREC_ACCURACY_PLACES,
                               msg='Discrepancy calculated incorrectly')

    def test_run_hybrid_reduced_to_zeros_fails(self):
        waterfall = commons.load_waterfall()

        nprofs = waterfall.shape[0]
        nparts = 50
        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]

        tomo = tmo.TomographyCpp(waterfall, xp.T)

        with self.assertRaises(expt.WaterfallReducedToZero,
                               msg='Waterfall getting reduced to zero '
                                   'should raise an Exception'):
            tomo.run_hybrid()

    def test_run_hybrid_xp_is_none_fails(self):
        nprofs = 5
        img_width = 10
        waterfall = np.ones((nprofs, img_width))
        tomo = tmo.TomographyCpp(waterfall)

        with self.assertRaises(expt.CoordinateError,
                               msg='Calling the run_hybrid function, with '
                                   'longitudinal_tomography.xp=None should '
                                   'raise an Exception'):
            tomo.run_hybrid()

    def test_run_hybrid_correct(self):
        waterfall = commons.load_waterfall()

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]

        tomo = tmo.TomographyCpp(waterfall)
        tomo.xp = xp.T
        weights = tomo.run_hybrid(1)

        cweights = np.array([0.00861438, 0.00880599, 0.00893367, 0.00909539,
                             0.00918903, 0.00922729, 0.00928258, 0.00928246,
                             0.00927822, 0.00929515, 0.00929504, 0.00921845,
                             0.0092014,  0.00918014, 0.00915893, 0.00914618,
                             0.00919728, 0.00912925, 0.00918038, 0.00924009,
                             0.00920603, 0.00930828, 0.00940621, 0.00953821,
                             0.00959356, 0.00969566, 0.00981058, 0.00988708,
                             0.00998924, 0.01007859, 0.01014255, 0.01023618,
                             0.01021914, 0.01028299, 0.01026588, 0.01021909,
                             0.01016806, 0.01015526, 0.01011276, 0.01008299,
                             0.00999782, 0.00992973, 0.00991267, 0.00985308,
                             0.00977646, 0.0097126,  0.00971697, 0.00967434,
                             0.00961899, 0.00961908])

        nptest.assert_almost_equal(
            weights, cweights, err_msg='Weights were calculated incorrectly')

    def test_run_hybrid_correct_diff(self):
        waterfall = commons.load_waterfall()

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]

        tomo = tmo.TomographyCpp(waterfall)
        tomo.xp = xp.T
        weights = tomo.run_hybrid(1)

        correct = 0.009561478717303548
        self.assertAlmostEqual(tomo.diff[0], correct,
                               msg='Discrepancy calculated incorrectly')

class TestTomographyGPU(unittest.TestCase):

    def setUp(self):
        try:
            import cupy as cp
        except ImportError:
            self.skipTest('CuPy not found - skipped tests')
        else:
            self.cp = cp
            conf.AppConfig.use_gpu()
            conf.AppConfig.set_double_precision()

    def tearDown(self) -> None:
        conf.AppConfig.use_cpu()
        conf.AppConfig.set_double_precision()


    def test_set_xp_not_iterable_fails(self):
        waterfall = self.cp.ones((10, 10))
        tomo = tmo.Tomography(waterfall)

        with self.assertRaises(expt.CoordinateImportError,
                               msg='Providing x coordinates as non iterable '
                                   'should raise an Exception'):
            tomo.xp = 1

    def test_set_xp_one_dim_fails(self):
        waterfall = self.cp.ones((10, 10))
        tomo = tmo.Tomography(waterfall)

        with self.assertRaises(expt.CoordinateImportError,
                               msg='Providing x coordinates array having '
                                   'another number of dimensions than two '
                                   'should raise an Exception'):
            tomo.xp = self.cp.array([1, 2, 4])

    def test_set_xp_wrong_nprofs_fails(self):
        nprofs = 5
        nbins = 10
        waterfall = self.cp.ones((nprofs, nbins))
        tomo = tmo.Tomography(waterfall)

        nparts = 20
        pts_nprofs = 2
        with self.assertRaises(expt.CoordinateImportError,
                               msg='Providing x coordinates array which has '
                                   'been tracked trough another number of '
                                   'profiles than found in the waterfall '
                                   'should raise an Exception'):
            tomo.xp = self.cp.ones((nparts, pts_nprofs))

    def test_set_xp_outside_of_img_upper_fails(self):
        nprofs = 5
        img_widthh = 10
        waterfall = self.cp.ones((nprofs, img_widthh))
        tomo = tmo.Tomography(waterfall)

        nparts = 20
        xp = self.cp.ones((nparts, nprofs))
        xp[1, 2] = img_widthh
        with self.assertRaises(expt.XPOutOfImageWidthError,
                               msg='Providing x coordinates array outside '
                                   'of image width should raise an Exception'):
            tomo.xp = xp

    def test_set_xp_outside_of_img_lower_fails(self):
        nprofs = 5
        img_width = 10
        waterfall = self.cp.ones((nprofs, img_width))
        tomo = tmo.Tomography(waterfall)

        nparts = 20
        xp = self.cp.ones((nparts, nprofs))
        xp[1, 2] = -1
        with self.assertRaises(expt.XPOutOfImageWidthError,
                               msg='Providing x coordinates array outside '
                                   'of image width should raise an Exception'):
            tomo.xp = xp

    def test_xp_can_be_set_to_none(self):
        nprofs = 5
        img_width = 10
        nparts = 20
        waterfall = self.cp.ones((nprofs, img_width))
        xp = self.cp.ones((nparts, nprofs))
        tomo = tmo.Tomography(waterfall, xp)
        tomo.xp = None

        self.assertEqual(
            tomo.xp, None, msg='xp should be able to be set to None')

    def test_waterfall_reduced_to_zeros_fail(self):
        nprofs = 5
        img_width = 10
        waterfall = -self.cp.ones((nprofs, img_width))

        with self.assertRaises(expt.WaterfallReducedToZero,
                               msg='Providing an all negative or '
                                   'zero waterfall should raise an Exception'):
            tomo = tmo.Tomography(waterfall)

    def test_run_xp_is_none_fails(self):
        nprofs = 5
        img_width = 10
        waterfall = self.cp.ones((nprofs, img_width))
        tomo = tmo.Tomography(waterfall)

        with self.assertRaises(expt.CoordinateError,
                               msg='Calling the run function, with '
                                   'longitudinal_tomography.xp=None '
                                   'should raise an Exception'):
            tomo.run()

    def test_run_correct_weights(self):
        waterfall = self.cp.array(commons.load_waterfall())

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = self.cp.meshgrid(self.cp.arange(0, nparts), self.cp.arange(nprofs))[0]

        tomo = tmo.Tomography(waterfall)
        tomo.xp = xp.T
        weights = tomo.run(1)

        cweights = np.array([0.00861438, 0.00880599, 0.00893367, 0.00909539,
                             0.00918903, 0.00922729, 0.00928258, 0.00928246,
                             0.00927822, 0.00929515, 0.00929504, 0.00921845,
                             0.0092014,  0.00918014, 0.00915893, 0.00914618,
                             0.00919728, 0.00912925, 0.00918038, 0.00924009,
                             0.00920603, 0.00930828, 0.00940621, 0.00953821,
                             0.00959356, 0.00969566, 0.00981058, 0.00988708,
                             0.00998924, 0.01007859, 0.01014255, 0.01023618,
                             0.01021914, 0.01028299, 0.01026588, 0.01021909,
                             0.01016806, 0.01015526, 0.01011276, 0.01008299,
                             0.00999782, 0.00992973, 0.00991267, 0.00985308,
                             0.00977646, 0.0097126,  0.00971697, 0.00967434,
                             0.00961899, 0.00961908])

        self.cp.testing.assert_array_almost_equal(
            weights, cweights, err_msg='Weights were calculated incorrectly on GPU')

    def test_run_correct_weights_singleprec(self):
        conf.AppConfig.set_single_precision()
        waterfall = self.cp.array(commons.load_waterfall(), dtype=self.cp.float32)

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = self.cp.meshgrid(self.cp.arange(0, nparts), self.cp.arange(nprofs))[0]

        tomo = tmo.Tomography(waterfall)
        tomo.xp = xp.T
        weights = tomo.run(1)

        cweights = np.array([0.00861438, 0.00880599, 0.00893367, 0.00909539,
                             0.00918903, 0.00922729, 0.00928258, 0.00928246,
                             0.00927822, 0.00929515, 0.00929504, 0.00921845,
                             0.0092014,  0.00918014, 0.00915893, 0.00914618,
                             0.00919728, 0.00912925, 0.00918038, 0.00924009,
                             0.00920603, 0.00930828, 0.00940621, 0.00953821,
                             0.00959356, 0.00969566, 0.00981058, 0.00988708,
                             0.00998924, 0.01007859, 0.01014255, 0.01023618,
                             0.01021914, 0.01028299, 0.01026588, 0.01021909,
                             0.01016806, 0.01015526, 0.01011276, 0.01008299,
                             0.00999782, 0.00992973, 0.00991267, 0.00985308,
                             0.00977646, 0.0097126,  0.00971697, 0.00967434,
                             0.00961899, 0.00961908])

        self.cp.testing.assert_array_almost_equal(
            weights, cweights, decimal=SINGLE_PREC_ACCURACY_PLACES, err_msg='Weights were calculated incorrectly on GPU')
        
    def test_run_gpu_cpu_equal(self):
        conf.AppConfig.use_cpu()
        waterfall = commons.load_waterfall()

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0]

        tomo = tmo.Tomography(waterfall)
        tomo.xp = xp.T
        weights = tomo.run(1)

        conf.AppConfig.use_gpu()

        waterfall_gpu = self.cp.array(commons.load_waterfall())
        waterfall_gpu = waterfall_gpu[:nprofs]
        waterfall_gpu = waterfall_gpu[:, 70:170]
        xp_gpu = self.cp.meshgrid(self.cp.arange(0, nparts), self.cp.arange(nprofs))[0]

        tomo_gpu = tmo.Tomography(waterfall_gpu)
        tomo_gpu.xp = xp_gpu.T
        weights_gpu = tomo_gpu.run(1)

        self.cp.testing.assert_array_almost_equal(
            weights, weights_gpu, err_msg='CPU and GPU weights do not match')

        self.cp.testing.assert_array_almost_equal(
            tomo.diff[0], tomo_gpu.diff[0], err_msg='CPU and GPU difference do not match')

    def test_run_correct_diff(self):
        waterfall = self.cp.array(commons.load_waterfall())

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = self.cp.meshgrid(self.cp.arange(0, nparts), self.cp.arange(nprofs))[0]

        tomo = tmo.Tomography(waterfall)
        tomo.xp = xp.T
        tomo.run(1)

        correct = 0.009561478717303546
        self.cp.testing.assert_array_almost_equal(tomo.diff[0], correct,
                               err_msg='Discrepancy calculated incorrectly on GPU')

    def test_run_correct_diff_singleprec(self):
        conf.AppConfig.set_single_precision()
        waterfall = self.cp.array(commons.load_waterfall(), dtype=self.cp.float32)

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = self.cp.meshgrid(self.cp.arange(0, nparts), self.cp.arange(nprofs))[0]

        tomo = tmo.Tomography(waterfall)
        tomo.xp = xp.T
        tomo.run(1)

        correct = 0.009561478717303546
        self.cp.testing.assert_array_almost_equal(tomo.diff[0], correct, decimal=SINGLE_PREC_ACCURACY_PLACES,
                               err_msg='Discrepancy calculated incorrectly on GPU')