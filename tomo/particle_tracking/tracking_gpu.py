# TEMP
import sys
import time as tm
# END TEMP

import numpy as np
# Better way of importing pycuda?
#   - Add check if connected before import?
#   - Use manual connection?
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray

from particle_tracking.tracking import ParticleTracker


class TrackPyCuda(ParticleTracker):

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

        dphi, denergy = self.calc_dphi_denergy(xp[0], yp[0])

        dphi = dphi.astype(np.float32)
        denergy = denergy.astype(np.float32)
        rf1v = rf1v.astype(np.float32)
        rf2v = rf2v.astype(np.float32)

        self._kick_and_drift(xp, yp, denergy, dphi, rf1v, rf2v,
                             nr_of_turns, nr_of_particles)


    def _kick_and_drift(self, xp, yp, denergy, dphi, rf1v, rf2v,
                        n_turns, n_part):
        tpar = self.timespace.par
        turn = 0
        denergy_gpu = gpuarray.to_gpu(denergy)
        dphi_gpu = gpuarray.to_gpu(dphi)

        nr_turns = 10000

        t0 = tm.perf_counter()
        for i in range(nr_turns):
            self._drift(dphi_gpu, denergy_gpu, tpar.dphase[turn])
        t1 = tm.perf_counter()
        print(t1 - t0)

        t0 = tm.perf_counter()
        for i in range(nr_turns):
            # dphi_gpu -= tpar.dphase[turn] * denergy_gpu
            dphi_gpu -= 0.4 * denergy_gpu
        t1 = tm.perf_counter()
        print(t1 - t0)

    def _drift(self, dphi_gpu, denergy_gpu, dphase):
        dphi_gpu -= dphase * denergy_gpu

