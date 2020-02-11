'''Module containing ParticleTracker class

:Author(s): **Christoffer Hjertø Grindheim**
'''

import numpy as np
import logging as log

from . import machine as mach
from . import particles as pts 
from ..utils import assertions as asrt
from ..utils import exceptions as expt


class ParticleTracker:
    '''Super class for tracking classes.
    
    The particle class holds some general utilities por particle tracking.
    These includes asserions of parameters and properties for flags.

    Parameters
    ----------
    machine: Machine
        Holds all information needed for particle tracking and generation of
        the particle distribution.

    Attributes
    ----------
    machine: Machine
        Holds all information needed for particle tracking and generation of
        the particle distribution.
    particles: Particles
        Creates and/or holds initial distribution of particles.
    nturns: int
        Number of machine turns particles should be tracked trough.
    self_field_flag: property, boolean
        Flag to indicate that self-fields should be included during tracking.
    fortran_flag: property, boolean
        Flag to indicate that the particle tracking should print fortran-style
        output strings to stdout during tracking.

    Raises
    ------
    MachineParameterError: Exception
        Input argument is not of type Machine, or Machine object\
        fields needed for tracking are missing.
    '''
    # The tracking routine works on a copy of the input coordinates. <- fix
    def __init__(self, machine):

        if not isinstance(machine, mach.Machine):
            err_msg = 'Input argument must be Machine.'
            raise expt.MachineParameterError(err_msg)

        self._assert_machine(machine)
        self.machine = machine
        self.particles = pts.Particles()

        self.nturns = machine.dturns * (machine.nprofiles - 1)
        self._self_field_flag = False
        self._ftn_flag = False

    @property
    def self_field_flag(self):
        '''self_field_flag defined as @property
        
        Flag can be set to true by calling the function: enable_self_fields.

        Returns
        -------
        self_field_flag: boolean
            Flag to indicate that self-fields are to be used during tracking.
        '''
        return self._self_field_flag
    
    @property
    def fortran_flag(self):
        '''self_field_flag defined as @property

        Flag can be set to true by calling the function: enable_fortran_output.

        Returns
        -------
        fortran_flag: boolean
            Flag to indicate that Fortran styled output is enabeled.
        '''
        return self._ftn_flag

    def enable_fortran_output(self, profile_charge):
        '''Function for enabeling of Fortran-styled output.
        
        Call this function in order to produce a Fortran-styled\
        output during the particle tracking. The output will be\
        printed to stdout.

        The output will initially be a print of the plot info needed for\
        the tomoscope application. During the tracking, the output will
        describe between which profiles the particles are currently\
        being tracked between. Also, the number of 'lost particles' will be\
        printed. This is however **not a real measurement**,\
        but a static string needed due to changes in the tracking algorithm. 

        Parameters
        ----------
        profile_charge: float
            Total charge of profile.

        Raises
        ------
        ProfileChargeNotCalculated: Exception
            Needed field for enabeling Fortran output,\
            profile charge, is missing from Machine object.
        '''
        if profile_charge is None:
            err_msg = 'profile_charge is needed for fortran-style output'
            raise expt.ProfileChargeNotCalculated(err_msg)
        self._ftn_flag = True
        self._profile_charge = profile_charge
        log.info('Fortran style output for particle tracking enabled!')

    def enable_self_fields(self, profiles):
        '''Function for enabeling particle tracking using self-fields.

        Call this function to track the particles using self-fields.
        Note that the self-field tracking is **much slower** than\
        tracking without self-fields. 

        Parameters
        ----------
        profiles: Profiles
            Self-field calculations must have been performed in\
            the Profiles object prior to calling this function.

        Raises
        ------
        SelfFieldTrackingError: Exception
            Needed fields for tracking using self-fields missing\
            from Machine object.
        '''
        needed_fieds = ['phiwrap', 'vself']
        asrt.assert_fields(profiles, 'profiles', needed_fieds,
                           expt.SelfFieldTrackingError)

        self._phiwrap = profiles.phiwrap
        self._vself = profiles.vself
        self._self_field_flag = True
        log.info('Tracking using self fields enabled!')

    # Checks that the given machine object includes the nesscessary
    # variables to perform the tracking.
    # Does not check parameters for calculating using self-fields.
    def _assert_machine(self, machine):
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
