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
    typedef int32_t int_t;
#else
    typedef double real_t;
    typedef int64_t int_t;
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

extern "C"
__global__ void generate_sin_lut(int_t *lut,
                                 real_t x0,
                                 real_t x1,
                                 int_t G,
                                 int_t S){
    std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;
    
    real_t dx = (x1 - x0) / (G - 1);
    for (std::size_t i = tid; i < G; i += stride){
        real_t x = x0 + i*dx;
        lut[i] = sin(x) * S;
    }
}

extern "C"
__device__ int_t sin_fixed_point(int_t x_int,
                      int_t x0_int,
                      int_t dx_int,
                      const int_t *lut,
                      int_t G,
                      bool fail_silently = false){
    int_t idx = (x_int - x0_int) / dx_int;
    if (idx < 0){
        if (fail_silently){
            return lut[0];
        } else {
            printf("The given value (%d) is less then the lower bound (%d) for the look-up table.\n", (int)(x_int), (int)(x0_int));
        }
    } else if (idx >= G){
        if (fail_silently){
            return lut[G - 1];
        } else {
            printf("The given value (%d) is greater then the upper bound (%d) for the look-up table.\n", (int)(x_int), (int)(x0_int+dx_int*G));
        }
    }
    return lut[idx];
} 

// Calculates the entire process of the kick/drift loop up.
extern "C"
__global__ void kick_drift_up_turns_int(const int_t * __restrict__ dphi,
                                    const int_t * __restrict__ denergy,
                                    int_t * __restrict__ xp,
                                    int_t * __restrict__ yp,
                                    const int_t * __restrict__ drift_coef,
                                    const int_t * __restrict__ rfv1,
                                    const int_t * __restrict__ rfv2,
                                    const int_t * __restrict__ phi0,
                                    const int_t * __restrict__ phi12,
                                    const int_t hratio,
                                    const int_t nr_particles,
                                    const int_t * __restrict__ acc_kick,
                                    int_t turn,
                                    const int_t nturns,
                                    const int_t dturns,
                                    int_t profile,
                                    int_t S,
                                    int_t G,
                                    real_t x0,
                                    real_t x1,
                                    const int_t * __restrict__ lut) {
    std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;
    
    int_t x0_int = x0 * S;
    real_t dx = (x1 - x0) / (G - 1);
    int_t dx_int = dx * S;
    int_t curr_turn = turn;
    int_t curr_profile = profile;
    
    for (int i = tid; i < nr_particles; i += stride)
    {
        int_t current_dphi = dphi[i];
        int_t current_denergy = denergy[i];
        curr_turn = turn;
        curr_profile = profile;

        while (curr_turn < nturns)
        {

            current_dphi -= drift_coef[curr_turn] * current_denergy / S;
            curr_turn++;

            current_denergy += (rfv1[curr_turn-1] * sin_fixed_point(current_dphi + phi0[curr_turn-1], 
                                                                x0_int, dx_int, lut, G) / S
                              + rfv2[curr_turn-1] * sin_fixed_point(hratio * (current_dphi + phi0[curr_turn-1] - phi12[curr_turn-1]),
                                                                x0_int, dx_int, lut, G) / S
                              - acc_kick[curr_turn-1] / S);

            if (curr_turn % dturns == 0)
            {
                curr_profile++;
                xp[nr_particles * curr_profile + i] = current_dphi;
                yp[nr_particles * curr_profile + i] = current_denergy;
            }
        }
    }
}

// Calculates the entire process of the kick/drift loop down.
extern "C"
__global__ void kick_drift_down_turns_int(const int_t * __restrict__ dphi,
                                      const int_t * __restrict__ denergy,
                                      int_t * __restrict__ xp,
                                      int_t * __restrict__ yp,
                                      const int_t * __restrict__ drift_coef,
                                      const int_t * __restrict__ rfv1,
                                      const int_t * __restrict__ rfv2,
                                      const int_t * __restrict__ phi0,
                                      const int_t * __restrict__ phi12,
                                      const int_t hratio,
                                      const int_t nr_particles,
                                      const int_t * __restrict__ acc_kick,
                                      int_t turn,
                                      const int_t dturns,
                                      int_t profile,
                                      int_t S,
                                      int_t G,
                                      real_t x0,
                                      real_t x1,
                                      const int_t * __restrict__ lut) {
    std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;
    
    int_t x0_int = x0 * S;
    real_t dx = (x1 - x0) / (G - 1);
    int_t dx_int = dx * S;
    int_t curr_turn = turn;
    int_t curr_profile = profile;

    for (int i = tid; i < nr_particles; i += stride)
    {
        int_t current_dphi = dphi[i];
        int_t current_denergy = denergy[i];
        curr_turn = turn;
        curr_profile = profile;

        while (curr_turn > 0)
        {
            current_denergy -= (rfv1[curr_turn-1] * sin_fixed_point(current_dphi + phi0[curr_turn-1], 
                                                                x0_int, dx_int, lut, G) / S
                              + rfv2[curr_turn-1] * sin_fixed_point(hratio * (current_dphi + phi0[curr_turn-1] - phi12[curr_turn-1]),
                                                                x0_int, dx_int, lut, G) / S
                              - acc_kick[curr_turn-1] / S);
            
            curr_turn--;
            current_dphi += drift_coef[curr_turn] * current_denergy / S;

            if (curr_turn % dturns == 0)
            {
                curr_profile--;
                xp[nr_particles * curr_profile + i] = current_dphi;
                yp[nr_particles * curr_profile + i] = current_denergy;
            }
        }
    }
}