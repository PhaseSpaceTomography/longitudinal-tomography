"""Module containing Machine class for storing
machine and reconstruction parameters

:Author(s): **Christoffer Hjertø Grindheim**, **Anton Lu**
"""

import logging
from typing import Tuple

import numpy as np
from scipy import optimize, constants

from .. import assertions as asrt
from ..utils import physics
from .machine_base import MachineABC

log = logging.getLogger(__name__)

_machine_opts_def = {
    'vrf1dot': 0.0,
    'vrf2': 0.0,
    'vrf2dot': 0.0,
    'bdot': 0.0,
}


class Machine(MachineABC):
    """Class holding machine and reconstruction parameters.

    This class holds machine parameters and information about the measurements.
    Also, it holds settings for the reconstruction process.

    The Machine class and its values are needed for the original particle
    tracking routine. Its values are used for calculation of reconstruction
    area and info concerning the phase space, the distribution of particles,
    and the tracking itself. In addition to this, the machine object is needed
    for the generation of
    :class:`~longitudinal_tomography.data.profiles.Profiles` objects.

    See superclass for documentation about inherited class variables.

    To summarize, the Machine class must be used if a program resembling the
    original Fortran version is to be created.

    Parameters
    ----------
    dturns: int
        Number of machine turns between each measurement.
    vrf1: float
        Peak voltage of the first RF system at the machine reference frame.
    mean_orbit_rad: float
        Mean orbit radius of machine [m].
    bending_rad: float
        Machine bending radius [m].
    b0: float
        B-field at machine reference frame [T].
    bdot: float
        Time derivative of B-field (considered constant) [T/s].
    trans_gamma: float
        Transitional gamma.
    e_rest: float
        Rest energy of accelerated particle [eV/C^2], saved as e_rest.
    nprofiles: int
        Number of measured profiles.
    nbins: int
        Number of bins in a profile.
    synch_part_x: float
        Synchronous phase given in number of bins, counting\
        from the lower profile bound to the synchronous phase.
    dtbin: float
        Size of profile bins [s].
    kwargs:
        All scalar attributes can be set via the kwargs.

    Attributes
    ----------
    vrf1: float
        Peak voltage of the first RF system at the machine reference frame.
    vrf2: float, default=0.0
        Peak voltage of the second RF system at the machine reference frame.\n
    vrf1dot: float, default=0.0
        Time derivatives of the voltages of the first RF system
        (considered constant).\n
    vrf2dot: float, default=0.0
        Time derivatives of the voltages of the second RF system
        (considered constant).\n
    b0: float
        B-field at machine reference frame [T].
    bdot: float
        Time derivative of B-field (considered constant) [T/s].
    h_num: int, default=1
        Principle harmonic number.\n
    """

    def __init__(self, dturns: int, vrf1: float, mean_orbit_rad: float,
                 bending_rad: float, b0: float, trans_gamma: float,
                 rest_energy: float, nprofiles: int, nbins: int, dtbin: float,
                 vat_now: bool = True, **kwargs):
        super().__init__(dturns, mean_orbit_rad, bending_rad, trans_gamma,
                         rest_energy, nprofiles, nbins, dtbin, **kwargs)

        kwargs_processed = super()._process_kwargs(_machine_opts_def, kwargs)

        # TODO: Take rfv info as a single input
        # TODO: Take b-field info as a single input

        # Machine parameters
        self.vrf1 = vrf1
        self.vrf1dot = kwargs_processed['vrf1dot']
        self.vrf2 = kwargs_processed['vrf2']
        self.vrf2dot = kwargs_processed['vrf2dot']
        self.b0 = b0
        self.bdot = kwargs_processed['bdot']

        self.fitted_synch_part_x = None
        self.bunchlimit_low = None
        self.bunchlimit_up = None

        if vat_now:
            self.values_at_turns()

    def values_at_turns(self):
        asrt.assert_machine_input(self)
        all_turns = (self.nprofiles - 1) * self.dturns

        # Create all-zero arrays of size nturns+1
        self._init_arrays(all_turns)

        # Calculate initial values at the machine reference frame (i0).
        i0 = self._array_initial_values()

        # Calculate remaining values for every machine turn.
        for i in range(i0 + 1, all_turns + 1):
            self.time_at_turn[i] = (self.time_at_turn[i - 1]
                                    + 2 * np.pi * self.mean_orbit_rad
                                    / (self.beta0[i - 1] * constants.c))

            try:
                self.phi0[i] = optimize.newton(func=physics.rf_voltage_mch,
                                               x0=self.phi0[i - 1],
                                               fprime=physics.drf_voltage_mch,
                                               tol=0.0001,
                                               maxiter=100,
                                               args=(self, i))
            except RuntimeError:
                raise ValueError('Could not fit synchronous phase for the '
                                 'supplied parameters.')

            self.e0[i] = (self.e0[i - 1]
                          + self.q
                          * physics.rf_voltage_at_phase(
                        self.phi0[i], self.vrf1, self.vrf1dot,
                        self.vrf2, self.vrf2dot, self.h_ratio,
                        self.phi12, self.time_at_turn, i))

            self.beta0[i] = np.sqrt(
                1.0 - (self.e_rest / float(self.e0[i])) ** 2)
            self.deltaE0[i] = self.e0[i] - self.e0[i - 1]
        for i in range(i0 - 1, -1, -1):
            self.e0[i] = (self.e0[i + 1]
                          - self.q
                          * physics.rf_voltage_at_phase(
                        self.phi0[i + 1], self.vrf1, self.vrf1dot,
                        self.vrf2, self.vrf2dot, self.h_ratio,
                        self.phi12, self.time_at_turn, i + 1))

            self.beta0[i] = np.sqrt(1.0 - (self.e_rest / self.e0[i]) ** 2)
            self.deltaE0[i] = self.e0[i + 1] - self.e0[i]

            self.time_at_turn[i] = (self.time_at_turn[i + 1]
                                    - 2 * np.pi * self.mean_orbit_rad
                                    / (self.beta0[i] * constants.c))

            try:
                self.phi0[i] = optimize.newton(func=physics.rf_voltage_mch,
                                               x0=self.phi0[i + 1],
                                               fprime=physics.drf_voltage_mch,
                                               tol=0.0001,
                                               maxiter=100,
                                               args=(self, i))
            except RuntimeError:
                raise ValueError('Could not fit synchronous phase for the '
                                 'supplied parameters.')

        # Calculate phase slip factor at each turn
        self.eta0 = physics.phase_slip_factor(self)

        # Calculates dphase for each turn
        self.drift_coef = physics.find_dphase(self)

        # Calculate revolution frequency at each turn
        self.omega_rev0 = physics.revolution_freq(self)

        # Calculate RF-voltages at each turn
        self.vrf1_at_turn, self.vrf2_at_turn = self._rfv_at_turns()

    def load_fitted_synch_part_x_ftn(self,
                                     fit_info: Tuple[float, float, float]):
        """Function for setting the synch_part_x if a fit has been performed.
        Saves parameters retrieved from the fitting routine
        needed by the
        :func:`longitudinal_tomography.compat.fortran.write_plotinfo`
        function in the
        :mod:`longitudinal_tomography.utils.tomo_output`. All needed info
        will be returned from the
        :func:`longitudinal_tomography.data.data_treatment.fit_synch_part_x`
        function.

        Sets the following fields:

        * fitted_synch_part_x
            The new x-coordinate of the synchronous particle (needed for
            :func:`longitudinal_tomography.compat.fortran.write_plotinfo`).
        * bunchlimit_low
            Lower phase of bunch (needed for
            :func:`longitudinal_tomography.compat.fortran.write_plotinfo`).
        * bunchlimit_up
            Upper phase of bunch (needed for
            :func:`longitudinal_tomography.compat.fortran.write_plotinfo`).
        * synch_part_x
            The x-coordinate of the synchronous particle.

        Parameters
        ----------
        fit_info: tuple
            Tuple should hold the following info in the following format:
            (F, L, U), where F is the fitted value of the synchronous particle,
            L is the lower bunch limit, and U is the upper bunch limit. All
            of the values should be given in bins. The info needed by the
            :func:`longitudinal_tomography.compat.fortran.write_plotinfo`
            function if a fit has been performed, and the a Fortran style
            output is to be given during the particle tracking.
        """
        log.info('Saving fitted synch_part_x to machine object.')
        self.fitted_synch_part_x = fit_info[0]
        self.bunchlimit_low = fit_info[1]
        self.bunchlimit_up = fit_info[2]
        self.synch_part_x = self.fitted_synch_part_x

    # Initiating arrays in order to store information about parameters
    # that has a different value every turn.
    def _init_arrays(self, all_turns: int):
        array_length = all_turns + 1
        self.time_at_turn = np.zeros(array_length)
        self.omega_rev0 = np.zeros(array_length)
        self.phi0 = np.zeros(array_length)
        self.drift_coef = np.zeros(array_length)
        self.deltaE0 = np.zeros(array_length)
        self.beta0 = np.zeros(array_length)
        self.eta0 = np.zeros(array_length)
        self.e0 = np.zeros(array_length)

    # Calculating start-values for the parameters that changes for each turn.
    # The reference frame where the start-values
    # are calculated is the machine reference frame.
    def _array_initial_values(self) -> int:
        i0 = self.machine_ref_frame * self.dturns
        self.time_at_turn[i0] = 0
        self.e0[i0] = physics.b_to_e(self)
        self.beta0[i0] = physics.lorentz_beta(self, i0)
        phi_lower, phi_upper = physics.find_phi_lower_upper(self, i0)
        # Synchronous phase of a particle on the nominal orbit
        self.phi0[i0] = physics.find_synch_phase_mch(
            self, i0, phi_lower, phi_upper)
        return i0

    # Using a linear approximation to calculate the RF voltage for each turn.
    def _rfv_at_turns(self) -> Tuple[np.ndarray, np.ndarray]:
        rf1v = self.vrf1 + self.vrf1dot * self.time_at_turn
        rf2v = self.vrf2 + self.vrf2dot * self.time_at_turn
        return rf1v, rf2v
