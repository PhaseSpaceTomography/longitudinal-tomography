"""Unit-tests for the physics module.

Run as python test_post_process.py in console or via coverage
"""
from __future__ import annotations
import typing as t
import unittest

import pytest

import longitudinal_tomography.shortcuts as shortcuts
import longitudinal_tomography.data.data_treatment as treat
import longitudinal_tomography.data.post_process as post_process
from .. import commons

if t.TYPE_CHECKING:
    from longitudinal_tomography.data.profiles import Profiles
    from longitudinal_tomography.tomography import Tomography
    from longitudinal_tomography.tracking import Machine
    from longitudinal_tomography.utils.tomo_input import Frames

# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = commons.get_machine_args()


@pytest.fixture(scope='module')
def tomography_params(
    machine_frames_profiles: t.Tuple[Machine, Frames, Profiles]
) -> t.Tuple[Tomography, Machine]:
    """Track and reconstruct once per module from the shared session data."""
    machine, _, profiles = machine_frames_profiles
    xp, yp = shortcuts.track(machine, 0)
    tomo = shortcuts.tomogram(profiles.waterfall, xp, yp, 2)
    return tomo, machine


class TestPostProcess(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _inject_fixtures(
        self, tomography_params: t.Tuple[Tomography, Machine]
    ) -> None:
        self.tomo, self.machine = tomography_params

    def test_emittance_rms(self):
        tomo, machine = self.tomo, self.machine

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)

        emittance_rms = post_process.emittance_rms(phase_space, t_bins, e_bins)

        self.assertAlmostEqual(emittance_rms, 0.06802901508017638)

    def test_emittance_90(self):
        tomo, machine = self.tomo, self.machine

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)

        emittance_90 = post_process.emittance_90(phase_space, t_bins, e_bins)

        self.assertAlmostEqual(emittance_90, 0.269939386246181)

    def test_emittance_fractional(self):
        tomo, machine = self.tomo, self.machine

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)

        emittance = post_process.emittance_fractional(phase_space, t_bins,
                                                      e_bins, fraction=80)

        self.assertAlmostEqual(emittance, 0.2183279430331099)

    def test_emittance_fractional_bounds(self):
        tomo, machine = self.tomo, self.machine

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)

        with self.assertRaises(ValueError,
                               msg='A fraction below 0 should raise an '
                                   'exception'):
            post_process.emittance_fractional(phase_space, t_bins,
                                              e_bins, fraction=-1)

        with self.assertRaises(ValueError,
                               msg='A fraction above 100 should raise an '
                                   'exception'):
            post_process.emittance_fractional(phase_space, t_bins,
                                              e_bins, fraction=101)

    def test_rms_dpp(self):
        tomo, machine = self.tomo, self.machine

        energy = 1098272089.0462158
        momentum = 570830158.7660657
        mass = 938272088.1604904

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)

        rms_dpp = post_process.rms_dpp(phase_space.sum(0), e_bins,
                                       energy, mass)
        self.assertAlmostEqual(rms_dpp, 0.0005028782274293471)

    def test_post_process(self):
        tomo, machine = self.tomo, self.machine

        energy = 1098272089.0462158
        momentum = 570830158.7660657
        mass = 938272088.1604904

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)
        processed_values = post_process.post_process(phase_space, t_bins,
                                                     e_bins, energy, mass)

        correct = {
            'emittance_rms': 0.06802901508017638,
            'emittance_90': 0.269939386246181,
            'rms_dp/p': 0.0005028782274293471,
        }

        for k, v in correct.items():
            self.assertAlmostEqual(v, processed_values[k])
