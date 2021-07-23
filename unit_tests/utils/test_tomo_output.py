"""Unit-tests for the tomo_output module.

Run as python test_tomo_output.py in console or via coverage
"""

import os
import shutil
import unittest
from unittest.mock import patch

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
        measured_profile = waterfall[:, 0] / waterfall[:, 0].sum()

        tout.show(phase_space, measured_profile, tomo.diff)
