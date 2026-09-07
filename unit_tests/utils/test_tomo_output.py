"""Unit-tests for the tomo_output module.

Run as python test_tomo_output.py in console or via coverage
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from typing import TYPE_CHECKING
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest
import pytest

import longitudinal_tomography.shortcuts as shortcuts
import longitudinal_tomography.utils.tomo_output as tout
import longitudinal_tomography.data.data_treatment as dtreat

if TYPE_CHECKING:
    from longitudinal_tomography.data.profiles import Profiles
    from longitudinal_tomography.tomography import Tomography
    from longitudinal_tomography.tracking import Machine
    from longitudinal_tomography.utils.tomo_input import Frames


@pytest.fixture(scope='module')
def tomography_params(
    machine_frames_profiles: tuple[Machine, Frames, Profiles]
) -> tuple[Tomography, Machine, Profiles]:
    """Track and reconstruct once per module from the shared session data."""
    machine, _, profiles = machine_frames_profiles
    xp, yp = shortcuts.track(machine, 0)
    tomo = shortcuts.tomogram(profiles.waterfall, xp, yp, 2)
    return tomo, machine, profiles


class TestTomoOut(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject_fixtures(
        self, tomography_params: tuple[Tomography, Machine, Profiles]
    ) -> None:
        self.tomo, self.machine, self.profiles = tomography_params

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

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
        waterfall = self.profiles.waterfall
        rec_prof = 0

        phase_space = dtreat.phase_space(self.tomo, self.machine, rec_prof)[-1]
        measured_profile = (waterfall[rec_prof]
                            / waterfall[rec_prof].sum())

        self.addCleanup(plt.close, 'all')

        tout.show(phase_space, self.tomo.diff, measured_profile)

        mock_show.assert_called_once()

        axes = plt.gcf().axes
        self.assertEqual(
            len(axes), 4,
            msg='Reconstruction should be presented in four subplots')

        nptest.assert_array_equal(
            axes[3].get_lines()[0].get_ydata(), self.tomo.diff,
            err_msg='Discrepancy was not plotted in the convergence subplot')
        nptest.assert_array_equal(
            axes[1].get_lines()[1].get_ydata(), measured_profile,
            err_msg='Measured profile was not plotted alongside the '
                    'reconstructed profile')
