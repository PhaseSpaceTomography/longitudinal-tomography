"""Unit-tests for the physics module.

Run as python test_post_process.py in console or via coverage
"""

import unittest

import longitudinal_tomography.data.data_treatment as treat
import longitudinal_tomography.data.post_process as post_process
from .. import commons

# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = commons.get_machine_args()


class TestPostProcess(unittest.TestCase):
    def test_emittance_rms(self):
        tomo, machine = commons.get_tomography_params()

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)

        emittance_rms = post_process.emittance_rms(phase_space, t_bins, e_bins)

        self.assertAlmostEqual(emittance_rms, 0.06802901508017638)

    def test_emittance_90(self):
        tomo, machine = commons.get_tomography_params()

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)

        emittance_90 = post_process.emittance_90(phase_space, t_bins, e_bins)

        self.assertAlmostEqual(emittance_90, 0.269939386246181)

    def test_emittance_fractional(self):
        tomo, machine = commons.get_tomography_params()

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)

        emittance = post_process.emittance_fractional(phase_space, t_bins,
                                                      e_bins, fraction=80)

        self.assertAlmostEqual(emittance, 0.2183279430331099)

    def test_emittance_fractional_bounds(self):
        tomo, machine = commons.get_tomography_params()

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)

        with self.assertRaises(ValueError):
            post_process.emittance_fractional(phase_space, t_bins,
                                              e_bins, fraction=-1)

            post_process.emittance_fractional(phase_space, t_bins,
                                              e_bins, fraction=101)

    def test_rms_dpp(self):
        tomo, machine = commons.get_tomography_params()

        energy = 1098272089.0462158
        momentum = 570830158.7660657
        mass = 938272088.1604904

        t_bins, e_bins, phase_space = treat.phase_space(tomo, machine)

        rms_dpp = post_process.rms_dpp(phase_space.sum(0), e_bins,
                                       energy, mass)
        self.assertAlmostEqual(rms_dpp, 0.0005028782274293471)

    def test_post_process(self):
        tomo, machine = commons.get_tomography_params()

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
