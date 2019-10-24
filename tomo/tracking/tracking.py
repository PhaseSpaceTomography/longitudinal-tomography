import numpy as np
from numba import njit
import logging as log
from utils.tomo_io import OutputHandler as oh
from tracking.particle_tracker import ParticleTracker
from cpp_routines.tomolib_wrappers import kick, drift


class Tracking(ParticleTracker):

    def __init__(self, ts, mi):
        super().__init__(ts, mi)

    def track(self):
        nr_of_particles = self.find_nr_of_particles()

        xp = np.zeros((self.timespace.par.profile_count, nr_of_particles))
        yp = np.copy(xp)

        # Creating a homogeneous distribution of particles
        xp[0], yp[0] = self._initiate_points()

        # Calculating radio frequency voltages for each turn
        rf1v = (self.timespace.par.vrf1
                + self.timespace.par.vrf1dot
                * self.timespace.par.time_at_turn) * self.timespace.par.q
        rf2v = (self.timespace.par.vrf2
                + self.timespace.par.vrf2dot
                * self.timespace.par.time_at_turn) * self.timespace.par.q

        # Retrieving some numbers for array creation

        nr_of_turns = (self.timespace.par.dturns
                       * (self.timespace.par.profile_count - 1))

        dphi, denergy = self.coords_to_phase_and_energy(xp[0], yp[0])

        dphi = np.ascontiguousarray(dphi)
        denergy = np.ascontiguousarray(denergy)
        rf1v = np.ascontiguousarray(rf1v)
        rf2v = np.ascontiguousarray(rf2v)

        if self.timespace.par.self_field_flag:
            xp, yp = self.kick_and_drift_self(xp, yp, denergy, dphi,
                                              rf1v, rf2v, nr_of_turns,
                                              nr_of_particles)
        else:
            xp, yp = self.kick_and_drift(xp, yp, denergy, dphi,
                                         rf1v, rf2v, nr_of_turns,
                                         nr_of_particles)

        xp, yp, nr_lost_pts = self.filter_lost_paricles(xp, yp)
        
        log.info(f'Lost {nr_lost_pts} particles - '\
                 f'{(nr_lost_pts / nr_of_particles) * 100}%'\
                 f' of all particles')

        xp = np.ascontiguousarray(xp)
        yp = np.ascontiguousarray(yp)

        return xp, yp

    # Function for tracking particles, including self field voltages
    def kick_and_drift_self(self, xp, yp, denergy,
                            dphi, rf1v, rf2v, n_turns, n_part):
        tpar = self.timespace.par

        profile = 0
        turn = 0
        oh.print_tracking_status_ccc(profile)
        while turn < n_turns:
            dphi = drift(denergy, dphi, tpar.dphase, n_part, turn)
            
            turn += 1
            
            temp_xp = self.calc_xp_sf(dphi, tpar.phi0[turn],
                                      tpar.x_origin, tpar.h_num,
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
                oh.print_tracking_status_ccc(profile)

        return xp, yp

    @staticmethod
    @njit
    def calc_xp_sf(dphi, phi0, x_origin, h_num, omega_rev0, dtbin, phiwrap):
        temp_xp = (dphi + phi0 - x_origin * h_num * omega_rev0 * dtbin)
        temp_xp = ((temp_xp - phiwrap * np.floor(temp_xp / phiwrap))
                    / (h_num * omega_rev0 * dtbin))
        return temp_xp



    def kick_and_drift(self, xp, yp, denergy, dphi, rf1v, rf2v,
                       n_turns, n_part):
        turn = 0
        profile = 0
        oh.print_tracking_status_ccc(profile)
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
                               - self.timespace.par.x_origin)
                yp[profile] = (denergy / float(self.mapinfo.dEbin)
                               + self.timespace.par.yat0)
                oh.print_tracking_status_ccc(profile)
        return xp, yp

    def coords_to_phase_and_energy(self, xp, yp, turn=0):
        return super().coords_to_phase_energy(
                self.timespace.par, xp, yp, self.mapinfo.dEbin)
