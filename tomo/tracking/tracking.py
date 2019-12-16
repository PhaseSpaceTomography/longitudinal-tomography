import numpy as np
from numba import njit # <- to be removed soon!
import logging as log

from ..utils import tomo_output as tomoout 
from . import particle_tracker as ptracker
from ..cpp_routines import tomolib_wrappers as tlw

class Tracking(ptracker.ParticleTracker):

    # Input should be thee coordinates of the particles given in
    #  phase and energy.
    # The output is the particles position in phase and energy for 
    #  each of the turns where a profile measurment is performed.
    def __init__(self, machine):
        super().__init__(machine)

    # Initial coordinates must be given as phase-space coordinates.
    # The function also returns phase and energies as phase-space coordinates.
    # Phase is given as angle relative to the synchronous phase.
    def track(self, recprof, init_distr=None):
        
        if init_distr is None:
            log.info('Creating homogeneous distribution of particles.')
            self.particles.homogeneous_distribution(self.machine, recprof)
            coords = self.particles.coordinates_dphi_denergy

            # Print fortran style plot info. Needed for tomograph.
            # Limits in I (phase), and dEbin must have been calculated.
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
            log.info('Tracking particles... (Self-fields enabled)')
            denergy = np.ascontiguousarray(denergy)
            dphi = np.ascontiguousarray(dphi)
            xp, yp = self.kick_and_drift_self(
                        denergy, dphi, rfv1, rfv2, recprof)
        else:
            nparts = len(dphi)
            nturns = self.machine.dturns * (self.machine.nprofiles - 1)
            xp = np.zeros((self.machine.nprofiles, nparts))
            yp = np.zeros((self.machine.nprofiles, nparts))
    
            xp, yp = tlw.kick_and_drift(
                        xp, yp, denergy, dphi, rfv1, rfv2, recprof,
                        nturns, nparts, machine=self.machine)
            
        log.info('Tracking completed!')
        return xp, yp

    # Input:
    #   - denergy: particle energy relative to synchronous particle [eV]
    #   - dphi: particle phase relative to synchronous particle [s]
    #   - rf1v: Radio frequency voltages 2, multiplied by particle charge
    #   - rf2v: Radio frequency voltages 2, multiplied by particle charge
    #   - rec_prof: Time slice to be reconstructed (scalar:int)
    # Output:
    #   - Energy for each particle at each measured time slice [eV]  
    #     Relative to synch. particle.
    #   - Phase (dphi) for each particle at each measured time slice [s]
    #     Relative to synch. particle.
    #   - Homogeneous distribution of particles is placed at rec_prof
    #     time-slice.
    def kick_and_drift(self, denergy, dphi, rf1v, rf2v, rec_prof):
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


    # Input:
    #   - denergy: particle energy relative to synchronous particle [eV]
    #   - dphi: particle phase relative to synchronous phase [rad]
    #   - rf1v: Radio frequency voltages 2, multiplied by particle charge
    #   - rf2v: Radio frequency voltages 2, multiplied by particle charge
    #   - rec_prof: Time slice to be reconstructed (scalar:int)
    # Output:
    #   - Energy for each particle at each measured time slice  
    #     Relative to synch. particle, given as bins in ps coordinate system.
    #   - Phase for each particle at each measured time slice [s]
    #     Relative to synch. particle, given as bins in ps coordinate system.
    #   - Homogeneous distribution of particles is placed at rec_prof
    #     time-slice.
    def kick_and_drift_self(self, denergy, dphi, rf1v, rf2v,
                            rec_prof):
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

        yp[rec_prof] = denergy / self.particles.dEbin + self.machine.yat0

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
                               + self.machine.yat0)
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
                               + self.machine.yat0)
                iprof -= 1

        return xp, yp


    # Calculate from physical coordinates to x-coordinates.
    # Needed for tracking using self-fields.
    # Will be converted to C++ befor you know it! 
    @staticmethod
    @njit
    def _calc_xp_sf(dphi, phi0, xorigin, h_num, omega_rev0, dtbin, phiwrap):
        temp_xp = (dphi + phi0 - xorigin * h_num * omega_rev0 * dtbin)
        temp_xp = ((temp_xp - phiwrap * np.floor(temp_xp / phiwrap))
                    / (h_num * omega_rev0 * dtbin))
        return temp_xp
