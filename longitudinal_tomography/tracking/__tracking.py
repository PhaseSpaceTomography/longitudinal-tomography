"""Module containing ParticleTracker class,
a super class for particle trackers.

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""
import logging
from typing import TYPE_CHECKING

from . import particles as pts
from .machine_base import MachineABC
from .. import assertions as asrt, exceptions as expt

if TYPE_CHECKING:
    from ..data.profiles import Profiles


log = logging.getLogger(__name__)


class ParticleTracker:
    """Super class for classes meant tracking particles.

    This class holds some general utilities for tracking particles. These
    utilities includes assertions and flags.

    Parameters
    ----------
    machine: MachineABC
        Holds all information needed for particle tracking and generating
        the particle distribution.

    Attributes
    ----------
    machine: MachineABC
        Holds all information needed for particle tracking and generation of
        the particle distribution.
    particles: Particles
        Creates and/or stores the initial distribution of particles.
    nturns: int
        Number of machine turns of which the particles should
        be tracked trough.
    _self_field_flag: boolean
        Flag to indicate that self-fields should be included
        during the tracking.
    _ftn_flag: boolean
        Flag to indicate that the particle tracking should print Fortran-style
        output strings to stdout during tracking.

    Raises
    ------
    MachineParameterError: Exception
        Input argument is not
        :class:`~longitudinal_tomography.tracking.machine.Machine`, or the
        Machine object provided is missing needed fields.
    """

    def __init__(self, machine: MachineABC):

        if not isinstance(machine, MachineABC):
            err_msg = 'Input argument must be Machine.'
            raise expt.MachineParameterError(err_msg)

        self._assert_machine(machine)
        self.machine = machine
        self.particles = pts.Particles()

        self.nturns = machine.dturns * (machine.nprofiles - 1)
        self._self_field_flag = False
        self._ftn_flag = False

        self._profile_charge = None
        self._phiwrap = None
        self._vself = None

    @property
    def self_field_flag(self):
        """self_field_flag defined as @property

        Flag can be set to true by calling :func:`enable_self_fields`.

        Returns
        -------
        self_field_flag: boolean
            Flag to indicate if particle tracking using
            self-fields is enabled.
        """
        return self._self_field_flag

    @property
    def fortran_flag(self):
        """self_field_flag defined as @property

        Flag can be set to true by calling :func:`enable_fortran_output`.

        Returns
        -------
        fortran_flag: boolean
            Flag to indicate if Fortran styled output is enabled.
        """
        return self._ftn_flag

    def enable_fortran_output(self, profile_charge: float):
        """Function for enabling of Fortran-styled output.

        Call this function in order to print a Fortran-styled output
        to stdout during the particle tracking.

        The output will initially be a print of the plot info needed for
        the tomoscope application. Then, During the tracking, the output will
        print which profiles the particles are currently being tracked
        between.

        The number of 'lost particles' will be also be printed.
        This is however **not a real measurement**, but a static string needed
        for the interface to the tomoscope application.
        The lost particles is not found during the tracking due to changes in
        the algorithm.

        Parameters
        ----------
        profile_charge: float
            Total charge of a reference profile.

        Raises
        ------
        ProfileChargeNotCalculated: Exception
            Needed field for enabling Fortran output,
            profile_charge, is missing from the Machine object.
        """
        if profile_charge is None:
            err_msg = 'profile_charge is needed for fortran-style output'
            raise expt.ProfileChargeNotCalculated(err_msg)
        self._ftn_flag = True
        self._profile_charge = profile_charge
        log.info('Fortran style output for particle tracking enabled!')

    def enable_self_fields(self, profiles: 'Profiles'):
        """Function for enabling particle tracking using self-fields.

        Call this function to track the particles using self-fields.
        Note that the self-field tracking is **much slower** than
        tracking without self-fields.

        Parameters
        ----------
        profiles: Profiles
            Self-fields must be calculated in the the Profiles object prior
            to calling this function.
            See :func:`longitudinal_tomography.data.profiles.Profiles.calc_self_fields`.

        Raises
        ------
        SelfFieldTrackingError: Exception
            Needed fields for tracking using self-fields missing\
            from Machine object.
        """
        needed_fieds = ['phiwrap', 'vself']
        asrt.assert_fields(profiles, 'profiles', needed_fieds,
                           expt.SelfFieldTrackingError)

        self._phiwrap = profiles.phiwrap
        self._vself = profiles.vself
        self._self_field_flag = True
        log.info('Tracking using self fields enabled!')

    # Checks that the given machine object includes the necessary
    # variables to perform the tracking.
    # Does not check parameters for calculating using self-fields.
    def _assert_machine(self, machine: MachineABC):
        needed_fieds = ['vrf1_at_turn', 'vrf2_at_turn', 'q',
                        'nprofiles', 'drift_coef', 'dturns', 'phi0',
                        'phi12', 'h_ratio', 'deltaE0', 'synch_part_x']
        asrt.assert_fields(
            machine, 'machine', needed_fieds, expt.MachineParameterError,
            'Did you remember to use machine.values_at_turns()?')
        asrt.assert_greater_or_equal(
            machine.synch_part_x, 'synch_part_x',
            0, expt.MachineParameterError,
            'particle tracking needs a valid synch_part_x value.')
