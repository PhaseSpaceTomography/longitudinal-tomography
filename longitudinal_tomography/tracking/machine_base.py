"""Module containing the abstract machine class

:Author(s): **Anton Lu**
"""
import typing as t
from abc import ABC, abstractmethod
import numpy as np
import logging

log = logging.getLogger(__name__)

_machine_base_defaults = {
    'demax': -1.E6,
    'phi12': 0.0,
    'h_ratio': 1.0,
    'synch_part_x': -1,
    'charge': 1,
    'h_num': 1,
    'g_coupling': None,
    'zwall_over_n': None,
    'min_dt': None,
    'max_dt': None,
    'self_field_flag': False,
    'full_pp_flag': False,
    'pickup_sensitivity': None,
    'machine_ref_frame': 0,
    'beam_ref_frame': 0,
    'snpt': 4,
    'niter': 20,
    'filmstart': 0,
    'filmstop': 1,
    'filmstep': 1,
    'output_dir': None,
}


class MachineABC(ABC):
    """Class holding machine and reconstruction parameters.

    ** NB: This is an abstract base class providing a skeleton for different
    implementations of the Fortran Machine class. **

    This class holds machine parameters and information about the measurements.
    Also, it holds settings for the reconstruction process.

    The Machine class and its values are needed for the original particle
    tracking routine. Its values are used for calculation of reconstruction
    area and info concerning the phase space, the distribution of particles,
    and the tracking itself. In addition to this, the machine object is needed
    for the generation of :class:`~tomo.data.profiles.Profiles` objects.

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
    demax: float, default=-1.E6
        Maximum energy of reconstructed phase space.\n
    dturns: int
        Number of machine turns between each measurement.
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
    mean_orbit_rad: float
        Mean orbit radius of machine [m].
    bending_rad: float
        Machine bending radius [m].
    b0: float
        B-field at machine reference frame [T].
    bdot: float
        Time derivative of B-field (considered constant) [T/s].
    phi12: float, default=0.0
        Phase difference between the two RF systems (considered constant).\n
    h_ratio: float, default=1.0
        Ratio of harmonics between the two RF systems.\n
    h_num: int, default=1
        Principle harmonic number.\n
    trans_gamma: float
        Transitional gamma.
    e_rest: float
        Rest energy of accelerated particle [eV/C^2].
    q: int, default=1
        Charge state of accelerated particle.\n
    g_coupling: float, default=None
        Space charge coupling coefficient (geometrical coupling coefficient).\n
    zwall_over_n: float, default=None
        Magnitude of Zwall/n, reactive impedance.\n
    min_dt: float, default=None
        Minimum phase of reconstruction area measured in seconds.\n
    max_dt: float, default=None
        Maximum phase of reconstruction area measured in seconds.\n
    nprofiles: int
        Number of measured profiles.
    pickup_sensitivity: float, default=None
        Effective pick-up sensitivity
        (in digitizer units per instantaneous Amp).\n
    nbins: int
        Number of bins in a profile.
    synch_part_x: float
        Synchronous phase given in number of bins, counting
        from the lower profile bound to the synchronous phase.
    synch_part_y: float
        Energy coordinate of synchronous particle in phase space coordinates
        of bins. The coordinate is set to be one half of the image width
        (nbins).
    dtbin: float
        Size of profile bins [s].
    dEbin: float
        Size of profile bins in energy.
    self_field_flag: boolean, default=False
        Flag to include self-fields in the tracking.\n
    full_pp_flag: boolean, default=False
        If set, all pixels in reconstructed phase space will be tracked.\n
    machine_ref_frame: int, default=0
        Frame to which machine parameters are referenced.\n
    beam_ref_frame: int, default=0
        Frame to which beam parameters are referenced.\n
    snpt: int, default=4
        Square root of particles pr. cell of phase space.\n
    niter: int, default=20
        Number of iterations in tomographic reconstruction.\n
    filmstart: int, default=0
        First profile to reconstruct.\n
    filmstop: int, default=1
        Last profile to reconstruct.\n
    filmstep: int, default=1
        Step between profiles to reconstruct.\n
    output_dir: string, default=None
        Directory to save output.\n
    time_at_turn: ndarray
        1D array holding the time at each turn. Turn zero = 0 [s].
    phi0: ndarray
        1D array holding the synchronous phase angle at the end of each turn.
    e0: ndarray
        1D array holding the total energy of synchronous
        particle at the end of each turn.
    beta0: ndarray
        1D array holding the Lorenz beta factor (v/c)
        at the end of each turn.
    deltaE0: ndarray
        1D array holding the difference between
        e0(n) and e0(n-1) for each turn.
    eta0: ndarray
        1D array holding the phase slip factor at each turn.
    drift_coef: ndarray
        1D array holding coefficient used for calculating difference,
        from phase n to phase n + 1. Needed by trajectory height
        calculator, and tracking.
    omega_rev0: ndarray
        1D array holding the revolution frequency at each turn.
    vrf1_at_turn: ndarray
        1D array holding the peak voltage at each turn for
        the first RF station.
    vrf2_at_turn: ndarray
        1D array holding the peak voltage at each turn for
        the second RF station.
    """

    def __init__(self, dturns: int, mean_orbit_rad: float, bending_rad: float,
                 trans_gamma: float, rest_energy: float,
                 n_profiles: int, n_bins: int, dtbin: float, **kwargs):
        processed_kwargs = self._process_kwargs(_machine_base_defaults, kwargs)

        self.dturns: int = dturns
        self.mean_orbit_rad: float = mean_orbit_rad
        self.bending_rad: float = bending_rad
        self.trans_gamma: float = trans_gamma
        self.e_rest: float = rest_energy

        self.nprofiles: int = n_profiles
        self.nbins: int = n_bins

        self.dtbin = dtbin

        # kwargs
        if processed_kwargs['min_dt'] is not None:
            self.min_dt = processed_kwargs['min_dt']
        else:
            self.min_dt = 0.0

        if processed_kwargs['max_dt'] is not None:
            self.max_dt = processed_kwargs['max_dt']
        else:
            self.max_dt = n_bins * dtbin

        self.demax: float = processed_kwargs['demax']
        self.phi12: t.Union[float, np.ndarray] = processed_kwargs['phi12']
        self.h_ratio: float = processed_kwargs['h_ratio']
        self.h_num: int = processed_kwargs['h_num']
        self.q: float = processed_kwargs['charge']
        self.g_coupling: float = processed_kwargs['g_coupling']
        self.zwall_over_n: float = processed_kwargs['zwall_over_n']
        self.synch_part_x: float = processed_kwargs['synch_part_x']

        # kwargs f:lags
        self.self_field_flag: bool = processed_kwargs['self_field_flag']
        self.full_pp_flag: bool = processed_kwargs['full_pp_flag']

        # kwargs reconstruction parameters
        self.machine_ref_frame: int = processed_kwargs['machine_ref_frame']
        self.beam_ref_frame: int = processed_kwargs['beam_ref_frame']
        self.pickup_sensitivity = processed_kwargs['pickup_sensitivity']
        self.snpt: int = processed_kwargs['snpt']
        self.niter: int = processed_kwargs['niter']
        self.filmstart: int = processed_kwargs['filmstart']
        self.filmstop: int = processed_kwargs['filmstop']
        self.filmstep: int = processed_kwargs['filmstep']
        self.output_dir: str = processed_kwargs['output_dir']

        # initialise attributes for later use
        # values at turns
        self.phi0: np.ndarray = None
        self.eta0: np.ndarray = None
        self.drift_coef: np.ndarray = None
        self.deltaE0: np.ndarray = None
        self.beta0: np.ndarray = None
        self.e0: np.ndarray = None
        self.omega_rev0: np.ndarray = None
        self.time_at_turn: np.ndarray = None
        self.vrf1_at_turn: np.ndarray = None
        self.vrf2_at_turn: np.ndarray = None

        # Used as flag for checking if particles particle tracking
        # has been done
        self.dEbin: float = None

    @property
    def nbins(self) -> int:
        """nbins defined as @property.
        Updates the position of the y-coordinate of the synchronous
        particle in the phase space coordinate system when set.

        Parameters
        ----------
        nbins: int
            Number of bins in a profile.

        Returns
        -------
        nbins: int
            Number of bins in a profile.
        """
        return self._nbins

    @nbins.setter
    def nbins(self, in_nbins: int):
        self._nbins = in_nbins
        self._find_synch_part_y()
        log.debug(f'synch_part_y was updated when the '
                  f'number of profile bins changed.\nNew values - '
                  f'nbins: {self._nbins}, synch_part_y: {self.synch_part_y}')

    # Function for asserting input dictionary for machine creator
    @classmethod
    def _process_kwargs(cls, defaults: t.Dict[str, t.Any], kwargs) -> t.Dict:
        use_params = {}

        for key in defaults:
            use_params[key] = defaults[key]

        for key in kwargs.copy():
            if key not in defaults:
                pass
                # raise KeyError(f'{key} is not a machine parameter')
            else:
                use_params[key] = kwargs.pop(key)
        return use_params

    @abstractmethod
    def values_at_turns(self):
        """Calculating machine values for each turn.

        The following values are calculated in this function. All are
        ndarrays of the data type float.

        * time_at_turn
            Time at each turn. Turn zero = 0 [s].
        * phi0
            Synchronous phase angle at the end of each turn.
        * e0
            Total energy of synchronous particle at the end of each turn.
        * beta0
            Lorentz beta factor (v/c) at the end of each turn.
        * deltaE0
            Difference between e0(n) and e0(n-1) for each turn.
        * eta0
            Phase slip factor at each turn.
        * drift_coef
            Coefficient used for calculating difference,
            from phase n to phase n + 1.
            Needed in trajectory height calculator and tracking.
        * omega_rev0
            Revolution frequency at each turn.
        * vrf1_at_turn
            Peak voltage at each turn for the first RF station.
        * vrf2_at_turn
            Peak voltage at each turn for the second RF station.

        The values are saved as fields of the Machine object.
        """
        pass

    # Function for finding y coordinate of synchronous particle in the
    # phase space coordinate system.
    def _find_synch_part_y(self):
        self.synch_part_y = self.nbins / 2.0
