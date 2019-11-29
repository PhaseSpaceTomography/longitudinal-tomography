import numpy as np
import logging as log

from ..new_machine import Machine
from ..particles import Particles
from ..utils import assertions as asrt
from ..utils import exceptions as expt
from ..utils import tomo_output as tomoout


class ParticleTracker:

    # The tracking routine works on a copy of the input coordinates.
    def __init__(self, machine):

        if not isinstance(machine, Machine):
            err_msg = 'Input argument must be Machine.'
            raise expt.MachineParameterError(err_msg)

        self._assert_machine(machine)
        self.machine = machine

        self.particles = Particles(self.machine)

        self.nturns = machine.dturns * (machine.nprofiles - 1)
        self._ftn_flag = False
        self._self_field_flag = False

    # Checks that the given machine object includes the nesscessary
    # variables to perform the tracking.
    # Does not check parameters for calculating using self-fields.
    def _assert_machine(self, machine):
        needed_fieds = ['vrf1_at_turn', 'vrf2_at_turn', 'q',
                        'nprofiles', 'dphase', 'dturns', 'phi0',
                        'phi12', 'h_ratio', 'deltaE0', 'xat0']
        asrt.assert_fields(
            machine, 'machine', needed_fieds, expt.MachineParameterError,
            'Did you remember to use machine.fill()?')
        asrt.assert_greater_or_equal(
            machine.xat0, 'xat0', 0, expt.MachineParameterError,
            'particle tracking needs a valid xat0 value.')

    # Only for Fortran output
    def enable_fortran_output(self, profile_charge):
        if profile_charge is None:
            err_msg = 'profile_charge is needed for fortran-style output'
            raise expt.ProfileChargeNotCalculated(err_msg)

        self._ftn_flag = True
        log.info('Fortran style output for particle tracking enabled!')
        print(tomoout.write_plotinfo_ftn(
                self.particles._psinfo, profile_charge))

    def enable_self_fields(self, profiles):
        needed_fieds = ['phiwrap', 'vself']
        asrt.assert_fields(profiles, 'profiles', needed_fieds,
                           expt.SelfFieldTrackingError)

        self._phiwrap = profiles.phiwrap
        self._vself = profiles.vself
        self._self_field_flag = True
        log.info('Tracking using self fields enabled!')



