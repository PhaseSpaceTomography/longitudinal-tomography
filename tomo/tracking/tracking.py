import numpy as np
from numba import njit
import logging as log
from utils.tomo_output import print_tracking_status_ftn 
from .particle_tracker import ParticleTracker
from cpp_routines.tomolib_wrappers import kick, drift


class Tracking(ParticleTracker):

    # Input should be thee coordinates of the particles given in
    #  phase and energy.
    # The output is the particles position in phase and energy for 
    #  each of the turns where a profile measurment is performed.
    def __init__(self, machine):
        super().__init__(machine)

    # Initial coordinates must be given as phase-space coordinates.
    # The function also returns phase and energies as phase-space coordinates.
    # Phase is given as difference in time (dt)
    def track(self, initial_coordinates=None, rec_prof=0):

        if initial_coordinates is None:
            log.info('Creating homogeneous distribution of particles.')
            self.particles.homogeneous_distribution()
        else:
            log.info('Using initial particle coordinates set by user.')
            self.particles.set_coordinates(initial_coordinates[0],
                                           initial_coordinates[1])

        rectrn = rec_prof * self.machine.dturns
        dphi0, denergy0 = self.particles.init_coords_to_physical(turn=rectrn)

        dphi = np.ascontiguousarray(dphi0)
        denergy = np.ascontiguousarray(denergy0)

        rfv1 = self.machine.vrf1_at_turn * self.machine.q
        rfv2 = self.machine.vrf2_at_turn * self.machine.q 

        # Tracking particles
        if self.machine.self_field_flag:
            raise NotImplementedError('kick and drift - '
                                      'self voltage not implemented (yet)')
            log.info('Tracking particles... (Self-fields enabled)')
            # xp, yp = self.kick_and_drift_self(
            #         xp, yp, denergy, dphi, rf1v, rf2v, nturns, nparts)
        else:
            log.info('Tracking particles... (Self-fields disabled)')
            (xp, yp) = self.kick_and_drift(denergy, dphi,
                                           rfv1, rfv2, rec_prof)
        log.info('Tracking completed!')

        xp, yp = self.particles.physical_to_coords(xp, yp)
        xp, yp, lost = self.particles.filter_lost_paricles(xp, yp)
        log.info(f'number of lost particles: {lost}')

        xp = xp.astype(np.int32).T
        yp = yp.astype(np.int32).T

        return xp, yp

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
            dphi = drift(denergy, dphi, self.machine.dphase,
                         nparts, turn)
            turn += 1
            # Calculating change in energy for each particle at a turn
            denergy = kick(self.machine, denergy, dphi, rf1v, rf2v,
                           nparts, turn)

            if turn % self.machine.dturns == 0:
                profile += 1
                out_dphi[profile] = np.copy(dphi)
                out_denergy[profile] = np.copy(denergy)
                if self._ftn_flag:
                    print_tracking_status_ftn(rec_prof, profile)

        # Starting again from homogeous distribution
        dphi = np.copy(out_dphi[rec_prof])
        denergy = np.copy(out_denergy[rec_prof])
        turn = rec_turn
        profile = rec_prof

        # Tracking 'downwards'
        while turn > 0:
            # Calculating change in energy for each particle at a turn
            denergy = kick(self.machine, denergy, dphi, rf1v, rf2v,
                           nparts, turn, up=False)
            turn -= 1
            # Calculating change in phase for each particle at a turn
            dphi = drift(denergy, dphi, self.machine.dphase,
                         nparts, turn, up=False)


            if turn % self.machine.dturns == 0:
                profile -= 1
                out_dphi[profile] = np.copy(dphi)
                out_denergy[profile] = np.copy(denergy)
                if self._ftn_flag:
                    print_tracking_status_ftn(rec_prof, profile)

        return out_dphi, out_denergy 


    # =========================== OLD ROUTINES ===============================
    #  To be deleted
    #
    # Optional tuple should contain the initial values
    #  of the particles coordinates
    #  which should be the start of the tracking
    #  (xp, yp)
    # If optional tuple is not provided, the particles
    #  will be tracked based on a homogeneous distribution of particles
    #  within the i and jlimits
    def _old_track(self, initial_coordinates=(), rec_prof=0, filter_lost=True):
        if len(initial_coordinates) > 0:
            # In this case, only the particles spescified by the user is tracked.
            # User input is checked for correctness before returning the values.
            ixp, iyp, nparts = self._manual_distribution(initial_coordinates)
        else:
            # In this case, a homogeneous distribution of particles is created
            #  within the i-jlimits.
            ixp, iyp, nparts = self._homogeneous_distribution()

        xp = np.zeros((self.timespace.par.nprofiles, nparts))
        yp = np.copy(xp)

        # Calculating radio frequency voltage multiplied by the
        #  particle charge at each turn.
        rf1v, rf2v = self.rfv_at_turns()

        # Calculating the number of turns of which
        #  the particles should be tracked through. 
        nturns = (self.timespace.par.dturns
                  * (self.timespace.par.nprofiles - 1))

        # Converting from coordinates to physical units
        dphi, denergy = self.coords_to_physical(ixp, iyp)

        # Ensuring that the memory is contigious for the C++ routines
        dphi = np.ascontiguousarray(dphi)
        denergy = np.ascontiguousarray(denergy)
        rf1v = np.ascontiguousarray(rf1v)
        rf2v = np.ascontiguousarray(rf2v)

        # Tracking particles
        if self.timespace.par.self_field_flag:
            xp, yp = self.kick_and_drift_self(
                    xp, yp, denergy, dphi, rf1v, rf2v, nturns, nparts)
        else:
            xp, yp = self.kick_and_drift(
                    xp, yp, denergy, dphi, rf1v, rf2v, nturns, nparts)

        # Setting initial values at the recreated profile.
        xp[rec_prof] = ixp
        yp[rec_prof] = iyp

        if filter_lost:
            xp, yp, nr_lost_pts = self.filter_lost_paricles(xp, yp)
        
            log.info(f'Lost {nr_lost_pts} particles - '\
                     f'{(nr_lost_pts / nparts) * 100}%'\
                     f' of all particles')

        xp = np.ascontiguousarray(xp)
        yp = np.ascontiguousarray(yp)

        return xp, yp

    # Function for tracking particles, including self field voltages
    def _old_kick_and_drift_self(self, xp, yp, denergy,
                            dphi, rf1v, rf2v, n_turns, n_part):
        tpar = self.timespace.par

        profile = 0
        turn = 0
        print_tracking_status_ftn(profile)
        while turn < n_turns:
            dphi = drift(denergy, dphi, tpar.dphase, n_part, turn)
            
            turn += 1
            
            temp_xp = self._calc_xp_sf(dphi, tpar.phi0[turn],
                                       self.timespace.par.xorigin, tpar.h_num,
                                       tpar.omega_rev0[turn], tpar.dtbin,
                                       tpar.phiwrap)

            selfvolt = self.timespace.vself[
                        profile,np.floor(temp_xp).astype(int)]

            denergy = kick(self.timespace.par, denergy, dphi,
                           rf1v, rf2v, n_part, turn)
            denergy += selfvolt * tpar.q

            if turn % tpar.dturns == 0:
                profile += 1
                xp[profile] = temp_xp
                yp[profile] = denergy / self.mapinfo.dEbin + tpar.yat0
                print_tracking_status_ftn(profile)

        return xp, yp

    def _old_kick_and_drift(self, xp, yp, denergy, dphi, rf1v, rf2v,
                       n_turns, n_part):
        turn = 0
        profile = 0
        print_tracking_status_ftn(profile)
        while turn < n_turns:
            # Calculating change in phase for each particle at a turn
            dphi = drift(denergy, dphi, self.timespace.par.dphase,
                         n_part, turn)
            turn += 1
            # Calculating change in energy for each particle at a turn
            denergy = kick(self.timespace.par, denergy, dphi, rf1v, rf2v,
                           n_part, turn)

            if turn % self.timespace.par.dturns == 0:
                profile += 1
                xp[profile] = ((dphi + self.timespace.par.phi0[turn])
                               / (float(self.timespace.par.h_num)
                               * self.timespace.par.omega_rev0[turn]
                               * self.timespace.par.dtbin)
                               - self.timespace.par.xorigin)
                yp[profile] = (denergy / float(self.mapinfo.dEbin)
                               + self.timespace.par.yat0)
                print_tracking_status_ftn(profile)
        return xp, yp

    @staticmethod
    @njit
    def _calc_xp_sf(dphi, phi0, xorigin, h_num, omega_rev0, dtbin, phiwrap):
        temp_xp = (dphi + phi0 - xorigin * h_num * omega_rev0 * dtbin)
        temp_xp = ((temp_xp - phiwrap * np.floor(temp_xp / phiwrap))
                    / (h_num * omega_rev0 * dtbin))
        return temp_xp

    # def coords_to_physical(self, xp, yp, turn=0):
    #     return super().coords_to_physical(
    #             self.timespace.par, xp, yp,
    #             self.mapinfo.dEbin,
    #             self.timespace.par.xorigin)

    # ======================== END OLD ROUTINES ============================
