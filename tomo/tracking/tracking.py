'''Module containing the Tracking class.

:Author(s): **Christoffer Hjertø Grindheim**
'''

from numba import njit
import numpy as np
import logging as log

from ..utils import tomo_output as tomoout 
from . import __tracking as ptracker
from ..cpp_routines import tomolib_wrappers as tlw

class Tracking(ptracker.ParticleTracker):
    '''Class for particle tracking.

    This class is needed in order to perform the particle tracking based on the
    algotithm from the original tomography program. Here, an initial
    distribution of test particles are homogeneously distributed across the
    reconstruction area. Later, these will be tracked trough all machine turns
    and saved for every time frame.

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
    '''
    def __init__(self, machine):
        super().__init__(machine)

    def track(self, recprof, init_distr=None):
        '''Primary function for tracking particles.

        The tracking routine starts at a given time frame with an initial
        distribution. From here, the particles are tracked 'forward' towards
        the last time frame and backwards towards the first time frame.
        The initial distribution should be placed on the time frame
        intended to be reconstructed.

        By default, the initial particle distribution will be homogeneously
        distributed over the reconstruction area. This is based on the 
        original Fortran tomography algorithm.
        An user spescified distribution can be given and override the 
        automatic particle generation.

        By calling the :py:meth:`~tomo.tracking.particle_tracker.
        ParticleTracker.enable_self_fields` function, a flag indicating that
        self-fields are to be included in the tracking is set. In this case,
        the :py:meth:`~tomo.tracking.tracking.Tracking.kick_and_drift_self`
        function will be used. Note that tracking including self fields
        are much slower than without.

        By calling the :py:meth:`~tomo.tracking.particle_tracker.
        ParticleTracker.enable_fortran_output` function, output resembling
        the original is written to stdout. Note that the values for the
        number of lost particles are **not valid**. Note also, if 
        the full Fortran output is to be printed, the automatic
        generation of particles has to be performed.
        
        Parameters
        ----------
        recprof: int
            Time frame to set as start-profile.
            Here the particle will have its initial distribution.
        init_distr: tuple, (ndarray, ndarray)
            An optional initial distributin. Must be given as a tuple of
            coordinates (dphi, denergy). dphi is the phase difference
            from the synchronous particle [rad]. denergy is the difference
            in energy from the synchronous particle.

        Returns
        -------
        xp: ndarray, float
            2D array containing each particles x-coordinate at
            each time frame. Array shape: (nprofiles, nparts).

            * If self-fields are enabeled,
              the coordinates will be given as phase-space coordinates.
            * If not, the returned x-coordinates will be given as
              phase [rad] relative to the synchronous particle.

        yp: ndarray, float
            2D array containing each particles y-coordinate at
            each time frame. Array shape: (nprofiles, nparts).

            * If self-fields are enabeled,
              the coordinates will be given as phase-space coordinates.
            * If not, the returned y-coordinates will be given as
              energy [eV] relative to the synchronous particle.

        '''
        if init_distr is None:
            # Homogeneous distribution is created based on the
            # original Fortran algorithm.
            log.info('Creating homogeneous distribution of particles.')
            self.particles.homogeneous_distribution(self.machine, recprof)
            coords = self.particles.coordinates_dphi_denergy

            # Print fortran style plot info. Needed for tomograph.
            if self.fortran_flag:
                print(tomoout.write_plotinfo_ftn(
                      self.machine, self.particles, self._profile_charge))
        
        else:
            log.info('Using initial particle coordinates set by user.')
            self.particles.coordinates_dphi_denergy = init_distr
            coords = self.particles.coordinates_dphi_denergy

        dphi = coords[0]
        denergy = coords[1]

        rfv1 = self.machine.vrf1_at_turn * self.machine.q
        rfv2 = self.machine.vrf2_at_turn * self.machine.q

        # Tracking particles
        if self.self_field_flag:
            # Tracking with self-fields
            log.info('Tracking particles... (Self-fields enabled)')
            denergy = np.ascontiguousarray(denergy)
            dphi = np.ascontiguousarray(dphi)
            xp, yp = self.kick_and_drift_self(
                        denergy, dphi, rfv1, rfv2, recprof)
        else:
            # Tracking without self-fields
            nparts = len(dphi)
            nturns = self.machine.dturns * (self.machine.nprofiles - 1)
            xp = np.zeros((self.machine.nprofiles, nparts))
            yp = np.zeros((self.machine.nprofiles, nparts))

            # Calling C++ implementation of tracking routine.
            xp, yp = tlw.kick_and_drift(
                        xp, yp, denergy, dphi, rfv1, rfv2, recprof,
                        nturns, nparts, machine=self.machine,
                        ftn_out=self.fortran_flag)
            
        log.info('Tracking completed!')
        return xp, yp

    def kick_and_drift(self, denergy, dphi, rf1v, rf2v, rec_prof):
        '''Routine for tracking a given distribution of particles.
        Implemented as hybrid between Python and C++. Kept for reference.
    
        Parameters
        ----------
        denergy: ndarray, float
            particle energy relative to synchronous particle [eV]
        dphi: ndarray, float
            particle phase relative to synchronous particle [s]
        rf1v: ndarray, float
            Radio frequency voltages (RF station 1),
            multiplied by particle charge
        rf2v: ndarray, float
            Radio frequency voltages (RF station 2),
            multiplied by particle charge
        rec_prof: int
            Time slice to initiate tracking.
            Initial particle distribution is placed here.

        Returns
        -------
        out_dphi:, ndarray, float
            phase [rad] for each particle at each measured time slice [s]
            Relative to the synchronous particle.
        out_denergy: ndarray, float
            Energy for each particle at each measured time slice [eV] 
            Relative to the synchronous particle.

        '''
        nparts = len(denergy)
        
        # Creating arrays for all tracked particles
        out_dphi = np.zeros((self.machine.nprofiles, nparts))
        out_denergy = np.copy(out_dphi)

        # Setting homogeneous coordinates to profile to be reconstructed.
        out_dphi[rec_prof] = np.copy(dphi)
        out_denergy[rec_prof] = np.copy(denergy)
        
        rec_turn = rec_prof * self.machine.dturns
        turn = rec_turn
        profile = rec_prof
        
        # Tracking 'upwards'
        while turn < self.nturns:
            # Calculating change in phase for each particle at a turn
            dphi = tlw.drift(denergy, dphi, self.machine.drift_coef,
                         nparts, turn)
            turn += 1
            # Calculating change in energy for each particle at a turn
            denergy = tlw.kick(self.machine, denergy, dphi, rf1v, rf2v,
                               nparts, turn)

            if turn % self.machine.dturns == 0:
                profile += 1
                out_dphi[profile] = np.copy(dphi)
                out_denergy[profile] = np.copy(denergy)
                if self.fortran_flag:
                    tomoout.print_tracking_status_ftn(rec_prof, profile)

        # Starting again from homogeous distribution
        dphi = np.copy(out_dphi[rec_prof])
        denergy = np.copy(out_denergy[rec_prof])
        turn = rec_turn
        profile = rec_prof

        # Tracking 'downwards'
        while turn > 0:
            # Calculating change in energy for each particle at a turn
            denergy = tlw.kick(self.machine, denergy, dphi, rf1v, rf2v,
                               nparts, turn, up=False)
            turn -= 1
            # Calculating change in phase for each particle at a turn
            dphi = tlw.drift(denergy, dphi, self.machine.drift_coef,
                             nparts, turn, up=False)


            if turn % self.machine.dturns == 0:
                profile -= 1
                out_dphi[profile] = np.copy(dphi)
                out_denergy[profile] = np.copy(denergy)
                if self.fortran_flag:
                    tomoout.print_tracking_status_ftn(rec_prof, profile)

        return out_dphi, out_denergy

    def kick_and_drift_self(self, denergy, dphi, rf1v, rf2v, rec_prof):
        '''Routine for tracking a given distribution of particles,\
        including self-fields. Implemented as hybrid between Python and C++.

        Used by the function :py:meth:`~tomo.tracking.tracking.Tracking.track`
        to track using self-fields.

        Returns the coordinates of the particles as phase space coordinates
        for efficiency reasons due to tracking algorithm based on the
        Fortran tomography. large room for improvement.
        
        Fortran output not yet supported.
        
        Parameters
        ----------
        denergy: ndarray, float
            particle energy relative to synchronous particle [eV]
        dphi: ndarray, float
            particle phase relative to synchronous particle [s]
        rf1v: ndarray, float
            Radio frequency voltages (RF station 1),
            multiplied by particle charge
        rf2v: ndarray, float
            Radio frequency voltages (RF station 2),
            multiplied by particle charge
        rec_prof: int
            Time slice to initiate tracking.
            Initial particle distribution is placed here.

        Returns
        -------
        xp: ndarray, float
            2D array holding the x-coordinates of each particles at
            each time frame (nprofiles, nparts). 
        yp: ndarray, float
            2D array holding the x-coordinates of each particles at
            each time frame (nprofiles, nparts).
            

        '''
        nparts = len(denergy)

        # Creating arrays for all tracked particles
        xp = np.zeros((self.machine.nprofiles, nparts))
        yp = np.copy(xp)

        rec_turn = rec_prof * self.machine.dturns
        turn = rec_turn
        iprof = rec_prof
        
        # To be saved for downwards tracking
        dphi0 = np.copy(dphi)
        denergy0 = np.copy(denergy)

        xp[rec_prof] = self._calc_xp_sf(
                            dphi, self.machine.phi0[rec_turn],
                            self.particles.xorigin,self.machine.h_num,
                            self.machine.omega_rev0[rec_turn],
                            self.machine.dtbin, self._phiwrap)

        yp[rec_prof] = (denergy / self.particles.dEbin
                        + self.machine.synch_part_y)

        print(iprof)
        while turn < self.nturns:
            dphi = tlw.drift(denergy, dphi, self.machine.drift_coef,
                             nparts, turn)      

            turn += 1

            temp_xp = self._calc_xp_sf(dphi, self.machine.phi0[turn],
                                       self.particles.xorigin,
                                       self.machine.h_num,
                                       self.machine.omega_rev0[turn],
                                       self.machine.dtbin,
                                       self._phiwrap)
            selfvolt = self._vself[iprof, temp_xp.astype(int) - 1]

            denergy = tlw.kick(self.machine, denergy, dphi, rf1v, rf2v,
                               nparts, turn)
            denergy += selfvolt * self.machine.q

            if turn % self.machine.dturns == 0:
                iprof += 1
                xp[iprof] = temp_xp
                yp[iprof] = (denergy / self.particles.dEbin
                               + self.machine.synch_part_y)
                print(iprof)


        dphi = dphi0
        denergy = denergy0
        turn = rec_turn
        iprof = rec_prof - 1
        temp_xp = xp[rec_prof]

        while turn > 0:
            selfvolt = self._vself[iprof, temp_xp.astype(int)]

            denergy = tlw.kick(self.machine, denergy, dphi, rf1v, rf2v,
                               nparts, turn, up=False)

            denergy += selfvolt * self.machine.q

            turn -= 1

            dphi = tlw.drift(denergy, dphi, self.machine.drift_coef,
                             nparts, turn, up=False) 

            temp_xp = self._calc_xp_sf(
                        dphi, self.machine.phi0[turn], self.particles.xorigin,
                        self.machine.h_num, self.machine.omega_rev0[turn],
                        self.machine.dtbin, self._phiwrap)

            if turn % self.machine.dturns == 0:
                print(iprof)
                xp[iprof] = temp_xp
                yp[iprof] = (denergy / self.particles.dEbin
                               + self.machine.synch_part_y)
                iprof -= 1

        return xp, yp


    # Calculate from physical coordinates to x-coordinates.
    # Needed for tracking using self-fields.
    # Will be converted to C++ before you know it! 
    @staticmethod
    @njit
    def _calc_xp_sf(dphi, phi0, xorigin, h_num, omega_rev0, dtbin, phiwrap):
        temp_xp = (dphi + phi0 - xorigin * h_num * omega_rev0 * dtbin)
        temp_xp = ((temp_xp - phiwrap * np.floor(temp_xp / phiwrap))
                    / (h_num * omega_rev0 * dtbin))
        return temp_xp
