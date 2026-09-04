"""Unit-tests for the tomo_output module.

Run as python test_tomo_output.py in console or via coverage
"""

import os
import shutil
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest

from .. import commons
import longitudinal_tomography.utils.tomo_output as tout
import longitudinal_tomography.data.data_treatment as dtreat

base_dir = os.path.split(os.path.realpath(__file__))[0]
base_dir = os.path.split(base_dir)[0]
tmp_dir = os.path.join(base_dir, 'tmp')


class TestTomoOut(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

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

    @patch('matplotlib.pyplot.show')
    def test_show(self, mock_show):
        _, _, profiles = commons.load_data()
        tomo, machine = commons.get_tomography_params()
        waterfall = profiles.waterfall
        rec_prof = 0

        phase_space = dtreat.phase_space(tomo, machine, rec_prof)[-1]
        measured_profile = (waterfall[rec_prof]
                            / waterfall[rec_prof].sum())

        self.addCleanup(plt.close, 'all')

        tout.show(phase_space, tomo.diff, measured_profile)

        mock_show.assert_called_once()

        axes = plt.gcf().axes
        self.assertEqual(
            len(axes), 4,
            msg='Reconstruction should be presented in four subplots')

        nptest.assert_array_equal(
            axes[3].get_lines()[0].get_ydata(), tomo.diff,
            err_msg='Discrepancy was not plotted in the convergence subplot')
        nptest.assert_array_equal(
            axes[1].get_lines()[1].get_ydata(), measured_profile,
            err_msg='Measured profile was not plotted alongside the '
                    'reconstructed profile')
