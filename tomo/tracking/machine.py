"""Module containing fundamental physics formulas

:Author(s): **Christoffer Hjertø Grindheim**
"""

import logging as log
import numpy as np
from scipy import optimize, constants

from ..utils import assertions as asrt
from ..utils import exceptions as expt
from ..utils import physics

_machine_opts_def = {}
_machine_opts_def['demax'] = -1.E6
_machine_opts_def['vrf1dot'] = 0.0 
_machine_opts_def['vrf2'] = 0.0
_machine_opts_def['vrf2dot'] = 0.0 
_machine_opts_def['phi12'] = 0.0
_machine_opts_def['h_ratio'] = 1.0
_machine_opts_def['h_num'] = 1 
_machine_opts_def['charge'] = 1
_machine_opts_def['g_coupling'] = None
_machine_opts_def['zwall_over_n'] = None
_machine_opts_def['min_dt'] = None 
_machine_opts_def['max_dt'] = None
_machine_opts_def['self_field_flag'] = False 
_machine_opts_def['full_pp_flag'] = False
_machine_opts_def['pickup_sensitivity'] = None 
_machine_opts_def['machine_ref_frame'] = 0 
_machine_opts_def['beam_ref_frame'] = 0 
_machine_opts_def['snpt'] = 4
_machine_opts_def['niter'] = 20
_machine_opts_def['filmstart'] = 0
_machine_opts_def['filmstop'] = 1
_machine_opts_def['filmstep'] = 1
_machine_opts_def['output_dir'] = None

default_opts = {}
def _reset_defaults():
    default_opts.update({key: _machine_opts_def[key]
                        for key in _machine_opts_def})
    for item in tuple(default_opts.keys()):
        if item not in _machine_opts_def:
            default_opts.pop(item)
_reset_defaults()

# Function for asserting input dictionary for machine creator
def _assert_machine_kwargs(**kwargs):
    use_params = {}

    for item in default_opts:
        use_params[item] = default_opts[item]

    for item in kwargs:
        if item not in default_opts:
            raise KeyError(f'{item} is not a machine parameter')
        else:
            use_params[item] = kwargs[item]
    return use_params

class Machine:
    '''Class holding machine and reconstruction parameters.
    
    This class holds machine parameters and information about the measurements.
    Also, it holds settings used to perform the reconstruction
    like then number of iterations of the tomorgaphic reconstruction.

    The Machine class is needed in order to perform the original particle
    tracking routine. In addition to this, it is needed for the generation
    of `Profiles` objects. Mostly this class is to be used if a program
    resembling the original Fortran version is to be created. 

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
    rest_energy: float
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
    demax: float
        Maximum energy of reconstructed phase space.\n
        Default value: -1.E6
    dturns: int
        Number of machine turns between each measurement.
    vrf1: float
        Peak voltage of the first RF system at the machine reference frame.
    vrf2: float
        Peak voltage of the second RF system at the machine reference frame.\n
        Default value: 0.0
    vrf1dot
        Time derivatives of the voltages of the first RF system\
        (considered constant).\n
        Default value: 0.0
    vrf2dot
        Time derivatives of the voltages of the second RF system\
        (considered constant).\n
        Default value: 0.0
    mean_orbit_rad: float
        Mean orbit radius of machine [m].
    bending_rad: float
        Machine bending radius [m].
    b0: float
        B-field at machine reference frame [T].
    bdot: float
        Time derivative of B-field (considered constant) [T/s].
    phi12: float
        Phase difference between the two RF systems (considered constant).\n
        Default value: 0.0
    h_ratio: float
        Ratio of harmonics between the two RF systems.\n
        Default value: 1.0
    h_num: int
        Principle harmonic number.\n
        Default value: 1
    trans_gamma: float
        Transitional gamma.
    e_rest: float
        Rest energy of accelerated particle [eV/C^2].
    q: int
        Charge state of accelerated particle.\n
        Default value: 1
    g_coupling: float
        Space charge coupling coefficient (geometrical coupling coefficient).\n
        Default value: None
    zwall_over_n: float
        Magnitude of Zwall/n, reactive impedance.\n
        Default value: None
    min_dt: float
        Minimum time in reconstruction area. <- fix\n
        Default value: None
    max_dt: float
        Maximum time in reconstruction area. <- fix\n
        Default value: None
    nprofiles: int
        Number of measured profiles.
    pickup_sensitivity: float
        Effective pick-up sensitivity\
        (in digitizer units per instantaneous Amp).\n
        Default value: None
    nbins: int
        Number of bins in a profile.
    synch_part_x: float
        Synchronous phase given in number of bins, counting\
        from the lower profile bound to the synchronous phase.
    dtbin: float
        Size of profile bins [s].
    self_field_flag: boolean
        Flag to include self-fields in the tracking.\n
        Default value: False
    full_pp_flag: boolean
        If set, all pixels in reconstructed phase space will be tracked.\n
        Default value: False
    machine_ref_frame: int
        Frame to which machine parameters are referenced.\n
        Default value: 0
    beam_ref_frame: int
        Frame to which beam parameters are referenced.\n
        Default value: 0
    snpt: int
        Square root of particles pr. cell of phase space.\n
        Default value: 4
    niter: int
        Number of iterations in tomographic reconstruction.\n
        Default value: 20
    filmstart: int
        First profile to reconstruct.\n
        Default value: 0
    filmstop: int
        Last profile to reconstruct.\n
        Default value: 1
    filmstep: int
        Step between profiles to reconstruct.\n
        Default value: 1
    output_dir: string
        Directory to save output.\n
        Default value: None
    time_at_turn: ndarray, float
        Time at each turn. Turn zero = 0 [s].
    phi0: ndarray, float
        Synchronous phase angle at the end of each turn.
    e0: ndarray, float
        Total energy of synchronous particle at the end of each turn.
    beta0: ndarray, float
        Lorenz beta factor (v/c) at the end of each turn.
    deltaE0: ndarray, float
        Difference between e0(n) and e0(n-1) for each turn.
    eta0: ndarray, float
        Phase slip factor at each turn.
    drift_coef: ndarray, float
        Coefficient used for calculating difference,\
        from phase n to phase n + 1.\
        Needed in trajectory height calculator and tracking.
    omega_rev0: ndarray, float
        Revolution frequency at each turn.
    vrf1_at_turn: ndarray, float
        Peak voltage at each turn for the first RF station.
    vrf1_at_turn: ndarray, float
        Peak voltage at each turn for the second RF station.
    '''

    def __init__(self, dturns, vrf1, mean_orbit_rad, bending_rad,
                 b0, bdot, trans_gamma, rest_energy, nprofiles, nbins,
                 synch_part_x, dtbin, **kwargs):

        kwargs = _assert_machine_kwargs(**kwargs)

        # TODO: Take rfv info as a single input
        # TODO: Take b-field info as a single input

        if kwargs['min_dt'] == None:
            min_dt = 0.0
        else:
            min_dt = kwargs['min_dt']

        if kwargs['max_dt'] == None:
            max_dt = nbins * dtbin
        else:
            max_dt = kwargs['max_dt']

        # Machine parameters
        self.demax = kwargs['demax']
        self.dturns = dturns
        self.vrf1 = vrf1
        self.vrf1dot = kwargs['vrf1dot']
        self.vrf2 = kwargs['vrf2']
        self.vrf2dot = kwargs['vrf2dot']
        self.mean_orbit_rad = mean_orbit_rad
        self.bending_rad = bending_rad
        self.b0 = b0
        self.bdot = bdot
        self.phi12 = kwargs['phi12']
        self.h_ratio = kwargs['h_ratio']
        self.h_num = kwargs['h_num']
        self.trans_gamma = trans_gamma
        self.e_rest = rest_energy
        self.q = kwargs['charge']
        self.g_coupling = kwargs['g_coupling']
        self.zwall_over_n = kwargs['zwall_over_n']
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.nprofiles = nprofiles
        self.pickup_sensitivity = kwargs['pickup_sensitivity']
        self.nbins = nbins
        self.synch_part_x = synch_part_x
        self.dtbin = dtbin

        # Flags
        self.self_field_flag = kwargs['self_field_flag']
        self.full_pp_flag = kwargs['full_pp_flag']

        # Reconstruction parameters
        self.machine_ref_frame = kwargs['machine_ref_frame']
        self.beam_ref_frame = kwargs['beam_ref_frame']
        self.snpt = kwargs['snpt']
        self.niter = kwargs['niter']
        self.filmstart = kwargs['filmstart']
        self.filmstop = kwargs['filmstop']
        self.filmstep = kwargs['filmstep']
        self.output_dir = kwargs['output_dir']

    @property
    def nbins(self):
        '''nbins defined as @property.
        Updates the position of the y coordinate of the synchronous\
        particle in the phase space coordinate system.

        Parameters
        ----------
        nbins: int
            Number of bins in a profile.

        Returns
        -------
        nbins: int
            Number of bins in a profile.
        '''
        return self._nbins

    @nbins.setter
    def nbins(self, in_nbins):
        self._nbins = in_nbins
        self._find_synch_part_y()
        log.info(f'synch_part_y was updated when the '
                 f'number of profile bins changed.\nNew values - '
                 f'nbins: {self.nbins}, synch_part_y: {self.synch_part_y}')

 
    def load_fitted_synch_part_x_ftn(self, fit_info):
        '''Function for setting the synch_part_x if a fit has been performed.
        Saves parameters retrieved from the fitting routine\
        needed by the `print_plotinfo` function in `tomo.utils.tomo_output`.

        Sets the following fields:

        * fitted_synch_part_x - The new x-coordinate of the\
                                synchronous particle\
                                (needed for `print_plotinfo`).
        * bunchlimit_low - Lower phase of bunch\
                           (needed for `print_plotinfo`).
        * bunchlimit_up - Upper phase of bunch\
                          (needed for `print_plotinfo`).
        * synch_part_x - The x-coordinate of the synchronous paricle.

        Parameters
        ----------
        fit_info: Tuple (fitted_synch_part_x, lower bunch limit,\
                        upper bunch limit)
            Info needed for the `print plotinfo` function if a fit\
            has been performed.
        '''
        log.info('Saving fitted synch_part_x to machine object.')
        self.fitted_synch_part_x = fit_info[0]
        self.bunchlimit_low = fit_info[1]
        self.bunchlimit_up = fit_info[2]
        self.synch_part_x = self.fitted_synch_part_x


    def values_at_turns(self):
        '''Calculating machine values for each turn.

        The following values are calculated in this function. All are
        ndarrays of the data type float.

        * time_at_turn - Time at each turn. Turn zero = 0 [s].
        * phi0 - Synchronous phase angle at the end of each turn.
        * e0 - Total energy of synchronous particle at the end of each turn.
        * beta0 - Lorenz beta factor (v/c) at the end of each turn.
        * deltaE0 - Difference between e0(n) and e0(n-1) for each turn.
        * eta0 - Phase slip factor at each turn.
        * drift_coef -  Coefficient used for calculating difference,\
                        from phase n to phase n + 1.\
                        Needed in trajectory height calculator and tracking.
        * omega_rev0 - Revolution frequency at each turn.
        * vrf1_at_turn - Peak voltage at each turn for the first RF station.
        * vrf2_at_turn - Peak voltage at each turn for the second RF station.

        The values are saved as fields of the Machine object.
        '''
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
         
            self.phi0[i] = optimize.newton(func=physics.rf_voltage,
                                           x0=self.phi0[i - 1],
                                           fprime=physics.drf_voltage,
                                           tol=0.0001,
                                           maxiter=100,
                                           args=(self, i))
            
            self.e0[i] = (self.e0[i - 1]
                          + self.q
                          * physics.short_rf_voltage_formula(
                                self.phi0[i], self.vrf1, self.vrf1dot,
                                self.vrf2, self.vrf2dot, self.h_ratio,
                                self.phi12, self.time_at_turn, i))

            self.beta0[i] = np.sqrt(1.0 - (self.e_rest/float(self.e0[i]))**2)
            self.deltaE0[i] = self.e0[i] - self.e0[i - 1]
        for i in range(i0 - 1, -1, -1):
            self.e0[i] = (self.e0[i + 1]
                          - self.q
                          * physics.short_rf_voltage_formula(
                                self.phi0[i + 1], self.vrf1, self.vrf1dot,
                                self.vrf2, self.vrf2dot, self.h_ratio,
                                self.phi12, self.time_at_turn, i + 1))

            self.beta0[i] = np.sqrt(1.0 - (self.e_rest/self.e0[i])**2)
            self.deltaE0[i] = self.e0[i + 1] - self.e0[i]

            self.time_at_turn[i] = (self.time_at_turn[i + 1]
                                    - 2 * np.pi * self.mean_orbit_rad
                                    / (self.beta0[i] * constants.c))
            
            self.phi0[i] = optimize.newton(func=physics.rf_voltage,
                                           x0=self.phi0[i + 1],
                                           fprime=physics.drf_voltage,
                                           tol=0.0001,
                                           maxiter=100,
                                           args=(self, i))

        # Calculate phase slip factor at each turn
        self.eta0 = physics.phase_slip_factor(self)

        # Calculates dphase for each turn
        self.drift_coef = physics.find_dphase(self)

        # Calculate revolution frequency at each turn
        self.omega_rev0 = physics.revolution_freq(self)

        # Calculate RF-voltages at each turn
        self.vrf1_at_turn, self.vrf2_at_turn = self._rfv_at_turns()

    # Initiating arrays in order to store information about parameters
    # that has a different value every turn.
    def _init_arrays(self, all_turns):
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
    def _array_initial_values(self):
        i0 = self.machine_ref_frame * self.dturns
        self.time_at_turn[i0] = 0
        self.e0[i0] = physics.b_to_e(self)
        self.beta0[i0] = physics.lorenz_beta(self, i0)
        phi_lower, phi_upper = physics.find_phi_lower_upper(self, i0)
        # Synchronous phase of a particle on the nominal orbit
        self.phi0[i0] = physics.find_synch_phase(
                            self, i0, phi_lower, phi_upper)
        return i0

    # Function for finding y coordinate of synchronous particle in the
    # phase space coordinate system.
    def _find_synch_part_y(self):
        self.synch_part_y = self.nbins / 2.0

    # Using a linear approximation to calculate the RF voltage for each turn.
    def _rfv_at_turns(self):
        rf1v = self.vrf1 + self.vrf1dot * self.time_at_turn
        rf2v = self.vrf2 + self.vrf2dot * self.time_at_turn
        return rf1v, rf2v
