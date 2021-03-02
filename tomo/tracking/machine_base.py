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
    """
    Base class for the Machine subclass
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
        self.q: float = processed_kwargs['charge']
        self.g_coupling: float = processed_kwargs['g_coupling']
        self.zwall_over_n: float = processed_kwargs['zwall_over_n']
        self.synch_part_x: float = processed_kwargs['synch_part_x']

        # kwargs f:lags
        self.self_field_flag: bool = processed_kwargs['self_field_flag']
        self.full_pp_flag: bool = processed_kwargs['full_pp_flag']

        # kwargs reconstruction parameters
        self.machine_ref_frame: float = processed_kwargs['machine_ref_frame']
        self.beam_ref_frame: float = processed_kwargs['beam_ref_frame']
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
        self.synch_part_y = self._nbins / 2.0
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
        pass
