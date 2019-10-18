import numpy as np
from numba import njit
from tracking.particle_tracker import ParticleTracker
from cpp_routines.tomolib_wrappers import kick, drift, kick_and_drift

# This is an experimental class, using a full c++ version
#   of the kick and drift routine.
# Not as thoroughly tested as the tracking.py version.
class TrackingCpp(ParticleTracker):

    def __init__(self, ts, mi):
        super().__init__(ts, mi)

    def track(self):
        # Experimental flag for calling GPU version of tracking routine.
        # Must find better way of deciding
        #   whether gpu routine(s) should be used or not.
        GPU = False
        
        tpar = self.timespace.par
        nr_of_particles = self.find_nr_of_particles()

        xp = np.zeros((tpar.profile_count, nr_of_particles))
        yp = np.copy(xp)

        # Creating a homogeneous distribution of particles
        xp[0], yp[0] = self._initiate_points()

        # Needed because of overwriting by GPU.
        if GPU:
            xp_0 = xp[0].copy()
            yp_0 = yp[0].copy()

        # Calculating radio frequency voltages for each turn
        rf1v = (tpar.vrf1
                + tpar.vrf1dot
                * tpar.time_at_turn) * tpar.q
        rf2v = (tpar.vrf2
                + tpar.vrf2dot
                * tpar.time_at_turn) * tpar.q

        nr_of_turns = (tpar.dturns
                       * (tpar.profile_count - 1))

        dphi, denergy = self.calc_dphi_denergy(xp[0], yp[0])

        dphi = np.ascontiguousarray(dphi)
        denergy = np.ascontiguousarray(denergy)
        rf1v = np.ascontiguousarray(rf1v)
        rf2v = np.ascontiguousarray(rf2v)
        xp = np.ascontiguousarray(xp)
        yp = np.ascontiguousarray(yp)

        if tpar.self_field_flag:
            xp, yp = self.kick_and_drift_self(xp, yp, denergy, dphi,
                                              rf1v, rf2v, nr_of_turns,
                                              nr_of_particles)
        else:
            xp, yp = kick_and_drift(xp, yp, denergy, dphi,
                                    rf1v, rf2v, tpar.phi0,
                                    tpar.deltaE0, tpar.omega_rev0,
                                    tpar.dphase, tpar.phi12, tpar.h_ratio,
                                    tpar.h_num, tpar.dtbin, tpar.x_origin,
                                    self.mapinfo.dEbin, tpar.yat0,
                                    tpar.dturns, nr_of_turns, nr_of_particles,
                                    gpu_flag=GPU)
        
        # Needed because of overwriting by GPU.
        if GPU:
            xp[0] = xp_0
            yp[0] = yp_0

        xp, yp, nr_lost_pts = self.filter_lost_paricles(xp, yp)
        
        print(f'Lost {nr_lost_pts} particles - '\
              f'{(nr_lost_pts / nr_of_particles) * 100}% of all particles')

        xp = np.ascontiguousarray(xp)
        yp = np.ascontiguousarray(yp)

        return xp, yp

    # Function for tracking particles, including self field voltages
    def kick_and_drift_self(self, xp, yp, denergy,
                            dphi, rf1v, rf2v, n_turns, n_part):
        tpar = self.timespace.par

        profile = 0
        turn = 0
        print(f'Tracking to profile {profile + 1}')
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
                print(f'Tracking to profile {profile + 1}')

        return xp, yp

    @staticmethod
    @njit
    def calc_xp_sf(dphi, phi0, x_origin, h_num, omega_rev0, dtbin, phiwrap):
        temp_xp = (dphi + phi0 - x_origin * h_num * omega_rev0 * dtbin)
        temp_xp = ((temp_xp - phiwrap * np.floor(temp_xp / phiwrap))
                    / (h_num * omega_rev0 * dtbin))
        return temp_xp

    def calc_dphi_denergy(self, xp, yp, turn=0):
        tpar = self.timespace.par
        dphi = ((xp + tpar.x_origin)
                * tpar.h_num
                * tpar.omega_rev0[turn]
                * tpar.dtbin
                - tpar.phi0[turn])
        denergy = (yp - tpar.yat0) * self.mapinfo.dEbin
        return dphi, denergy