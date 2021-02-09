"""Unit-tests for the Particles class.

Run as python test_particles_class.py in console or via coverage
"""

import unittest

from .. import commons
import longitudinal_tomography.tracking.machine as mch
import longitudinal_tomography.tracking.particles as pts
from longitudinal_tomography import exceptions as expt

# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = commons.get_machine_args()


class TestParticles(unittest.TestCase):

    def test_set_coordinates_dphi_denergy_valid(self):
        dphi = [0.2, 0.4, 0.5]
        denergy = [2142.4, 5432.2, 6732.764]
        input_coords = (dphi, denergy)
        ndims = len(input_coords)

        parts = pts.Particles()
        parts.coordinates_dphi_denergy = input_coords

        for i in range(ndims):
            for read, correct in zip(parts.coordinates_dphi_denergy[i],
                                     input_coords[i]):
                self.assertEqual(read, correct,
                                 msg='Something went wrong when setting '
                                     'coordinates in units of dphi and '
                                     'denergy')

    def test_set_coordinates_dphi_denergy_coord_tuple_no_iter_fails(self):
        input_coords = 1

        parts = pts.Particles()

        with self.assertRaises(
                expt.InvalidParticleError,
                msg='Non iterable input should raise an exception'):
            parts.coordinates_dphi_denergy = input_coords

    def test_set_coordinates_dphi_denergy_coord_one_coord_provided_fails(self):
        dphi = [0.2, 0.4, 0.5]
        input_coords = (dphi, )

        parts = pts.Particles()

        with self.assertRaises(
                expt.InvalidParticleError,
                msg='Providin only one coordinate should raise an exception'):
            parts.coordinates_dphi_denergy = input_coords

    def test_set_coordinates_dphi_denergy_coord_no_iter_fails(self):
        dphi = 0.2
        denergy = 2142.4
        input_coords = (dphi, denergy)

        parts = pts.Particles()

        with self.assertRaises(
                expt.InvalidParticleError,
                msg='Providing non iterable coordinates '
                    'should raise an exception'):
            parts.coordinates_dphi_denergy = input_coords

    def test_set_coordinates_dphi_denergy_coord_bad_len_fails(self):
        dphi = [0.2, 0.4, 0.5]
        denergy = [2142.4, 5432.2]
        input_coords = (dphi, denergy)

        parts = pts.Particles()

        with self.assertRaises(
                expt.InvalidParticleError,
                msg='Providing coordinate arrays of unequal length '
                    'should raise an exception'):
            parts.coordinates_dphi_denergy = input_coords

    def test_homogeneous_distribution_correct_ilimits(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        rbn = 10
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.synch_part_x /= float(rbn)

        parts = pts.Particles()
        parts.homogeneous_distribution(machine, recprof=21)

        imin_correct = 0
        imax_correct = 75

        self.assertEqual(parts.imin, imin_correct,
                         msg='imin has an unexpected value.')
        self.assertEqual(parts.imax, imax_correct,
                         msg='imax has an unexpected value.')

    def test_homogeneous_distribution_correct_jmin(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        rbn = 10
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.synch_part_x /= float(rbn)

        parts = pts.Particles()
        parts.homogeneous_distribution(machine, recprof=21)

        correct = [38, 38, 38, 38, 37, 31, 28, 26, 24, 22, 21, 19, 18, 17,
                   16, 15, 15, 14, 13, 13, 12, 11, 11, 11, 10, 10, 10, 9,
                   9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11,
                   11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 18, 18,
                   19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                   33, 35, 38]

        for j, corr in zip(parts.jmin, correct):
            self.assertEqual(j, corr, msg='Unexpected value found in jmin')

    def test_homogeneous_distribution_correct_jmax(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        rbn = 10
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.synch_part_x /= float(rbn)

        parts = pts.Particles()
        parts.homogeneous_distribution(machine, recprof=21)

        correct = [38, 38, 38, 38, 39, 45, 48, 50, 52, 54, 55, 57, 58, 59,
                   60, 61, 61, 62, 63, 63, 64, 65, 65, 65, 66, 66, 66, 67,
                   67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 66, 66,
                   66, 66, 65, 65, 65, 64, 64, 63, 63, 62, 62, 61, 60, 60,
                   59, 58, 58, 57, 56, 56, 55, 54, 53, 52, 51, 50, 49, 48,
                   47, 46, 45, 43, 41, 38]

        for j, corr in zip(parts.jmax, correct):
            self.assertEqual(j, corr, msg='Unexpected value found in jmin')

    def test_homogeneous_distribution_correct_phase(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        rbn = 250
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.synch_part_x /= float(rbn)

        parts = pts.Particles()
        parts.homogeneous_distribution(machine, recprof=21)

        correct = [-0.24180582, 0.04498875, 0.33178332, 0.61857789,
                   -0.24180582, 0.04498875, 0.33178332, 0.61857789,
                   -0.24180582, 0.04498875, 0.33178332, 0.61857789,
                   -0.24180582, 0.04498875, 0.33178332, 0.61857789]

        for phase, corr in zip(parts.coordinates_dphi_denergy[0], correct):
            self.assertAlmostEqual(
                phase, corr, msg='Error in setting of phase coordinate in '
                                 'cell of phase space at initial distribution')

    def test_homogeneous_distribution_correct_energy(self):
        machine = mch.Machine(**MACHINE_ARGS)
        machine.values_at_turns()

        rbn = 250
        machine.nbins = int(machine.nbins / rbn)
        machine.dtbin *= rbn
        machine.synch_part_x /= float(rbn)

        parts = pts.Particles()
        parts.homogeneous_distribution(machine, recprof=21)

        correct = [-115567.32591061, -115567.32591061, -115567.32591061,
                   -115567.32591061, -38522.4419702,   -38522.4419702,
                   -38522.4419702,   -38522.4419702,   38522.4419702,
                   38522.4419702,    38522.4419702,    38522.4419702,
                   115567.32591061,  115567.32591061,  115567.32591061,
                   115567.32591061]

        for energy, corr in zip(parts.coordinates_dphi_denergy[1], correct):
            self.assertAlmostEqual(
                energy, corr, msg='Error in setting of phase coordinate in '
                                  'cell of phase space at initial distribution'
            )
