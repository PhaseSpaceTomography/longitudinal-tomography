"""
Unit-tests for the shortcuts module

Run as python test_shortcuts.py in console or via coverage
"""
from __future__ import annotations
import unittest
import numpy as np
import numpy.testing as nptest

import longitudinal_tomography.tracking.machine as mch
import longitudinal_tomography.shortcuts as shortcuts
from . import commons


MACHINE_ARGS = commons.get_machine_args()


class TestShortcuts(unittest.TestCase):
    def test_track(self):

        machine = mch.Machine(**MACHINE_ARGS)

        # Tracking only a few particles for 20 time frames.
        machine.snpt = 1
        rbn = 13
        machine.rbn = rbn
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.nprofiles = 20

        machine.values_at_turns()

        xp, yp = shortcuts.track(machine, 10)

        # Comparing the coordinates of particle #0 only.
        correct_x = [8, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10,
                     11, 12, 12, 13, 14, 15, 4, 5, 6]

        correct_y = [14, 13, 13, 13, 12, 16, 16, 15, 15, 14,
                     14, 13, 13, 13, 12, 12, 11, 18, 18, 17]

        for x, cx in zip(xp[:, 0], correct_x):
            self.assertAlmostEqual(
                x, cx, msg='Error in tracking of particle '
                           'found in x-coordinate')
        for y, cy in zip(yp[:, 0], correct_y):
            self.assertAlmostEqual(
                y, cy, msg='Error in tracking of particle '
                           'found in y-coordinate')

    def test_tomogram(self):

        waterfall = commons.load_waterfall()

        nprofs = 10
        nparts = 50
        waterfall = waterfall[:nprofs]
        waterfall = waterfall[:, 70:170]
        xp = np.meshgrid(np.arange(0, nparts), np.arange(nprofs))[0].T

        tomo = shortcuts.tomogram(waterfall, xp, yp=None, n_iter=1)

        weights = tomo.weight
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
