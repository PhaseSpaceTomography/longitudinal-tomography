/**
 * @file kick_and_drift.cu
 *
 * @author Bernardo Abreu Figueiredo
 * Contact: bernardo.abreu.figueiredo@cern.ch
 *
 * CUDA kernels that handle particle tracking (kicking and
 * drifting.
 */

#ifdef USEFLOAT
    typedef float real_t;
#else
    typedef double real_t;
#endif


// Calculates the energy kick up for all the particles.
// This function does not iterate, so the
// amount of threads should be equal to nr_particles
extern "C"
__global__ void kick_up(const real_t * __restrict__ dphi,
                        real_t * __restrict__ denergy,
                        const real_t rfv1,
                        const real_t rfv2,
                        const real_t phi0,
                        const real_t phi12,
                        const real_t hratio,
                        const int nr_particles,
                        const real_t acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < nr_particles)
        denergy[tid] += rfv1 * sin(dphi[tid] + phi0)
                    + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick;
}

// Calculates the energy kick down for all the particles.
// This function does not iterate, so the
// amount of threads should be equal to nr_particles

extern "C"
__global__ void kick_down(const real_t * __restrict__ dphi,
                          real_t * __restrict__ denergy,
                          const real_t rfv1,
                          const real_t rfv2,
                          const real_t phi0,
                          const real_t phi12,
                          const real_t hratio,
                          const int nr_particles,
                          const real_t acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < nr_particles)
        denergy[tid] -= rfv1 * sin(dphi[tid] + phi0)
                    + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick;
}

// Calculates the phase drift up for all the particles.
// This function does not iterate, so the
// amount of threads should be equal to nr_particles
extern "C"
__global__ void drift_up(real_t * __restrict__ dphi,
                         const real_t * __restrict__ denergy,
                         const real_t drift_coef,
                         const int nr_particles) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < nr_particles)
        dphi[tid] -= drift_coef * denergy[tid];
}

// Calculates the phase drift down for all the particles.
// This function does not iterate, so the
// amount of threads should be equal to nr_particles
extern "C"
__global__ void drift_down(real_t * __restrict__ dphi,
                           const real_t * __restrict__ denergy,
                           const real_t drift_coef,
                           const int nr_particles) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < nr_particles)
        dphi[tid] += drift_coef * denergy[tid];
}

// Calculates the phase drift and energy kick up for all the particles.
// This function does not iterate, so the
// amount of threads should be equal to nr_particles
extern "C"
__global__ void kick_drift_up_simultaneously(real_t * __restrict__ dphi,
                                             real_t * __restrict__ denergy,
                                             const real_t drift_coef,
                                             const real_t rfv1,
                                             const real_t rfv2,
                                             const real_t phi0,
                                             const real_t phi12,
                                             const real_t hratio,
                                             const int nr_particles,
                                             const real_t acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < nr_particles)
    {
        dphi[tid] -= drift_coef * denergy[tid];
        denergy[tid] += rfv1 * sin(dphi[tid] + phi0)
                        + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick;
    }
}

// Calculates the phase drift and energy kick down for all the particles.
// This function does not iterate, so the
// amount of threads should be equal to nr_particles
extern "C"
__global__ void kick_drift_down_simultaneously(real_t * __restrict__ dphi,
                                               real_t * __restrict__ denergy,
                                               const real_t drift_coef,
                                               const real_t rfv1,
                                               const real_t rfv2,
                                               const real_t phi0,
                                               const real_t phi12,
                                               const real_t hratio,
                                               const int nr_particles,
                                               const real_t acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < nr_particles)
    {
        denergy[tid] -= (rfv1 * sin(dphi[tid] + phi0)
                        + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick);
        dphi[tid] += drift_coef * denergy[tid];
    }
}

// Calculates the entire process of the kick/drift loop up.
// This function does not iterate with respect to the amount of particles, so the
// amount of threads should be equal to nr_particles.
extern "C"
__global__ void kick_drift_up_turns(const real_t * __restrict__ dphi,
                                    const real_t * __restrict__ denergy,
                                    real_t * __restrict__ xp,
                                    real_t * __restrict__ yp,
                                    const real_t * __restrict__ drift_coef,
                                    const real_t * __restrict__ rfv1,
                                    const real_t * __restrict__ rfv2,
                                    const real_t * __restrict__ phi0,
                                    const real_t * __restrict__ phi12,
                                    const real_t hratio,
                                    const int nr_particles,
                                    const real_t * __restrict__ acc_kick,
                                    int turn,
                                    const int nturns,
                                    const int dturns,
                                    int profile) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < nr_particles)
    {
        real_t current_dphi = dphi[tid];
        real_t current_denergy = denergy[tid];

        while (turn < nturns)
        {

            current_dphi -= drift_coef[turn] * current_denergy;
            turn++;
            current_denergy += (rfv1[turn] * sin(current_dphi + phi0[turn])
                        + rfv2[turn] * sin(hratio * (current_dphi + phi0[turn] - phi12[turn])) - acc_kick[turn]);

            if (turn % dturns == 0)
            {
                profile++;
                xp[nr_particles * profile + tid] = current_dphi;
                yp[nr_particles * profile + tid] = current_denergy;
            }
        }
    }
}

// Calculates the entire process of the kick/drift loop down.
// This function does not iterate with respect to the amount of particles, so the
// amount of threads should be equal to nr_particles.
extern "C"
__global__ void kick_drift_down_turns(const real_t * __restrict__ dphi,
                                      const real_t * __restrict__ denergy,
                                      real_t * __restrict__ xp,
                                      real_t * __restrict__ yp,
                                      const real_t * __restrict__ drift_coef,
                                      const real_t * __restrict__ rfv1,
                                      const real_t * __restrict__ rfv2,
                                      const real_t * __restrict__ phi0,
                                      const real_t * __restrict__ phi12,
                                      const real_t hratio,
                                      const int nr_particles,
                                      const real_t * __restrict__ acc_kick,
                                      int turn,
                                      const int dturns,
                                      int profile) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < nr_particles)
    {
        real_t current_dphi = dphi[tid];
        real_t current_denergy = denergy[tid];

        while (turn > 0)
        {
            current_denergy -= (rfv1[turn] * sin(current_dphi + phi0[turn])
                        + rfv2[turn] * sin(hratio * (current_dphi + phi0[turn] - phi12[turn])) - acc_kick[turn]);
            turn--;
            current_dphi += drift_coef[turn] * current_denergy;

            if (turn % dturns == 0)
            {
                profile--;
                xp[nr_particles * profile + tid] = current_dphi;
                yp[nr_particles * profile + tid] = current_denergy;
            }
        }
    }
}