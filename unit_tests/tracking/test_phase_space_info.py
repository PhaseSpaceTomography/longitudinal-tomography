"""Unit-tests for the PhaseSpaceInfo class.

Run as python test_phase_space_info.py in console or via coverage
"""
from __future__ import annotations
import unittest

from .. import commons
import longitudinal_tomography.tracking.machine as mch
import longitudinal_tomography.tracking.phase_space_info as psi
from longitudinal_tomography import exceptions as expt

# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = commons.get_machine_args()


class TestMachine(unittest.TestCase):

    def test_find_dEbin_demax_lt_zero_vrf2_zero_output_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.xorigin = -246.60492626420734
        dEbin = psinfo.find_dEbin()

        correct_dEbin = 1232.7181430465346
        self.assertAlmostEqual(dEbin, correct_dEbin,
                               msg='dEbin was calculated incorrectly')

    def test_find_dEbin_demax_lt_zero_vrf2_not_zero_output_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.vrf2 = 1500
        machine.h_num = machine.vrf2 = 1500
        machine.h_num = 2
        machine.values_at_turns()

        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.xorigin = -290.3024631321037
        dEbin = psinfo.find_dEbin()

        correct_dEbin = 185.1099466310996
        self.assertAlmostEqual(dEbin, correct_dEbin,
                               msg='dEbin was calculated incorrectly')

    def test_find_dEbin_demax_gt_zero_vrf2_not_zero_output_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.demax = 1.0e7
        machine.values_at_turns()

        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.xorigin = -246.60492626420734
        dEbin = psinfo.find_dEbin()

        correct_dEbin = 26315.78947368421
        self.assertAlmostEqual(dEbin, correct_dEbin,
                               msg='dEbin was calculated incorrectly')

    def test_find_dEbin_demax_eq_zero_fails(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.demax = 0.0
        machine.values_at_turns()

        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.xorigin = -246.60492626420734

        with self.assertRaises(
                expt.EnergyBinningError,
                msg='demax equal to zero should raise exception'):
            dEbin = psinfo.find_dEbin()

    def test_find_dEbin_demax_xorigin_in_none_fails(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.demax = 0.0
        machine.values_at_turns()

        psinfo = psi.PhaseSpaceInfo(machine)

        with self.assertRaises(
                expt.EnergyBinningError,
                msg='xorigin with value None should raise exception.'):
            dEbin = psinfo.find_dEbin()

    def test_calc_xorigin_correct(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        psinfo = psi.PhaseSpaceInfo(machine)
        xorigin = psinfo.calc_xorigin()

        correct_xorigin = -246.60492626420734
        self.assertAlmostEqual(xorigin, correct_xorigin,
                               msg='xorigin was calculated incorrectly')

    def test_find_binned_phase_energy_limits_correct_ilims(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.find_binned_phase_energy_limits()

        correct_imin = 0
        correct_imax = 759

        self.assertEqual(psinfo.imin, correct_imin,
                         msg='imin was calculated incorrectly'
                             '(full_pp_flag not enabled)')
        self.assertEqual(psinfo.imax, correct_imax,
                         msg='imin was calculated incorrectly'
                             '(full_pp_flag not enabled)')

    def test_find_binned_phase_energy_limits_correct_jmax(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        rbn = 10
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.synch_part_x /= float(rbn)

        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.find_binned_phase_energy_limits()

        correct = [38, 38, 38, 38, 39, 45, 48, 50, 52, 54, 55, 57, 58, 59,
                   60, 61, 61, 62, 63, 63, 64, 65, 65, 65, 66, 66, 66, 67,
                   67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 66, 66,
                   66, 66, 65, 65, 65, 64, 64, 63, 63, 62, 62, 61, 60, 60,
                   59, 58, 58, 57, 56, 56, 55, 54, 53, 52, 51, 50, 49, 48,
                   47, 46, 45, 43, 41, 38]

        for j, corr in zip(psinfo.jmax, correct):
            self.assertEqual(j, corr,
                             msg='jmax calculated incorrectly'
                                 '(full_pp_flag not enabled)')

    def test_find_binned_phase_energy_limits_correct_jmin(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        rbn = 10
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.synch_part_x /= float(rbn)

        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.find_binned_phase_energy_limits()

        correct = [38, 38, 38, 38, 37, 31, 28, 26, 24, 22, 21, 19, 18, 17,
                   16, 15, 15, 14, 13, 13, 12, 11, 11, 11, 10, 10, 10, 9,
                   9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11,
                   11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 18, 18,
                   19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                   33, 35, 38]

        for j, corr in zip(psinfo.jmin, correct):
            self.assertEqual(j, corr,
                             msg='jmin calculated incorrectly'
                                 '(full_pp_flag not enabled)')

    def test_find_binned_phase_energy_limits_full_pp_correct_ilims(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        machine.full_pp_flag = True

        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.find_binned_phase_energy_limits()

        correct_imin = 0
        correct_imax = 759

        self.assertEqual(psinfo.imin, correct_imin,
                         msg='imin was calculated incorrectly'
                             '(full_pp_flag enabled)')
        self.assertEqual(psinfo.imax, correct_imax,
                         msg='imin was calculated incorrectly'
                             '(full_pp_flag enabled)')

    def test_find_binned_phase_energy_limits_full_pp_correct_jmax(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        machine.full_pp_flag = True

        rbn = 10
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.synch_part_x /= float(rbn)

        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.find_binned_phase_energy_limits()

        correct = [76] * machine.nbins

        for j, corr in zip(psinfo.jmax, correct):
            self.assertEqual(j, corr,
                             msg='jmax calculated incorrectly'
                                 '(full_pp_flag enabled)')

    def test_find_binned_phase_energy_limits_full_pp_correct_jmin(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        machine.full_pp_flag = True

        rbn = 10
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.synch_part_x /= float(rbn)

        psinfo = psi.PhaseSpaceInfo(machine)
        psinfo.find_binned_phase_energy_limits()

        correct = [1] * machine.nbins

        for j, corr in zip(psinfo.jmin, correct):
            self.assertEqual(j, corr,
                             msg='jmin calculated incorrectly'
                                 '(full_pp_flag enabled)')
