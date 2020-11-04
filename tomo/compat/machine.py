"""
Compatibility module. Everything in this submodule can be removed at once with
minimal impact to the rest of the package (some if-statements need to be
removed in the rest of the package as well).

The intention of this package is to provide Fortran I/O compatibility with the
tomoscope until it is deprecated.
"""
from typing import Tuple

import logging

from ..tracking.machine import Machine as SuperMachine


log = logging.getLogger(__name__)


class Machine(SuperMachine):
    """
    Machine object for use with Fortran. Intended to be for backwards compatibility.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialise all variables init
        self.fitted_synch_part_x = None
        self.bunchlimit_low = None
        self.bunchlimit_up = None

    def load_fitted_synch_part_x(self,
                                 fit_info: Tuple[float, float, float]):
        """Function for setting the synch_part_x if a fit has been performed.
        Saves parameters retrieved from the fitting routine
        needed by the :func:`tomo.utils.tomo_output.write_plotinfo_ftn`
        function in the :mod:`tomo.utils.tomo_output`. All needed info
        will be returned from the
        :func:`tomo.utils.data_treatment.fit_synch_part_x` function.

        Sets the following fields:

        * fitted_synch_part_x
            The new x-coordinate of the synchronous particle
            (needed for :func:`tomo.utils.tomo_output.write_plotinfo_ftn`).
        * bunchlimit_low
            Lower phase of bunch (needed for
            :func:`tomo.utils.tomo_output.write_plotinfo_ftn`).
        * bunchlimit_up
            Upper phase of bunch (needed for
            :func:`tomo.utils.tomo_output.write_plotinfo_ftn`).
        * synch_part_x
            The x-coordinate of the synchronous particle.

        Parameters
        ----------
        fit_info: tuple
            Tuple should hold the following info in the following format:
            (F, L, U), where F is the fitted value of the synchronous particle,
            L is the lower bunch limit, and U is the upper bunch limit. All
            of the values should be given in bins. The info needed by the
            :func:`tomo.utils.tomo_output.write_plotinfo_ftn` function
            if a fit has been performed, and the a Fortran style output is
            to be given during the particle tracking.
        """
        log.info('Saving fitted synch_part_x to machine object.')
        self.fitted_synch_part_x = fit_info[0]
        self.bunchlimit_low = fit_info[1]
        self.bunchlimit_up = fit_info[2]
        self.synch_part_x = self.fitted_synch_part_x


