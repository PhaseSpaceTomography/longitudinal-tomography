"""Unit-tests for the tomo_output module.

Run as python test_tomo_output.py in console or via coverage
"""

import os
import shutil
import unittest

import numpy as np
import numpy.testing as nptest

from .. import commons
import tomo.utils.tomo_output as tout

base_dir = os.path.split(os.path.realpath(__file__))[0]
base_dir = os.path.split(base_dir)[0]
tmp_dir = os.path.join(base_dir, 'tmp')

# All values retrieved from INDIVShavingC325.dat
MACHINE_ARGS = commons.get_machine_args()


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
