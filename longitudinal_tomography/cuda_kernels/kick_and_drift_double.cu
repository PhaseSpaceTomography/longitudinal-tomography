/**
 * @file kick_and_drift.cu
 *
 * @author Bernardo Abreu Figueiredo
 * Contact: bernardo.abreu.figueiredo@cern.ch
 *
 * CUDA kernels that handle particle tracking (kicking and
 * drifting).
 */


extern "C"
__global__ void kick_up(const double * __restrict__ dphi,
                        double * __restrict__ denergy,
                        const double rfv1,
                        const double rfv2,
                        const double phi0,
                        const double phi12,
                        const double hratio,
                        const int nr_particles,
                        const double acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < nr_particles)
        denergy[tid] += rfv1 * sin(dphi[tid] + phi0)
                    + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick;
}

extern "C"
__global__ void kick_down(const double * __restrict__ dphi,
                          double * __restrict__ denergy,
                          const double rfv1,
                          const double rfv2,
                          const double phi0,
                          const double phi12,
                          const double hratio,
                          const int nr_particles,
                          const double acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < nr_particles)
        denergy[tid] -= rfv1 * sin(dphi[tid] + phi0)
                    + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick;
}

extern "C"
__global__ void drift_up(double * __restrict__ dphi,
                         const double * __restrict__ denergy,
                         const double drift_coef,
                         const int nr_particles) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < nr_particles)
        dphi[tid] -= drift_coef * denergy[tid];
}

extern "C"
__global__ void drift_down(double * __restrict__ dphi,
                         const double * __restrict__ denergy,
                         const double drift_coef,
                         const int nr_particles) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < nr_particles)
        dphi[tid] += drift_coef * denergy[tid];
}

extern "C"
__global__ void kick_drift_up_simultaneously(double * __restrict__ dphi,
                         double * __restrict__ denergy,
                         const double drift_coef,
                         const double rfv1,
                         const double rfv2,
                         const double phi0,
                         const double phi12,
                         const double hratio,
                         const int nr_particles,
                         const double acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < nr_particles)
    {
        dphi[tid] -= drift_coef * denergy[tid];
        denergy[tid] += rfv1 * sin(dphi[tid] + phi0)
                        + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick;
    }
    
}

extern "C"
__global__ void kick_drift_down_simultaneously(double * __restrict__ dphi,
                         double * __restrict__ denergy,
                         const double drift_coef,
                         const double rfv1,
                         const double rfv2,
                         const double phi0,
                         const double phi12,
                         const double hratio,
                         const int nr_particles,
                         const double acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < nr_particles)
    {
        denergy[tid] -= (rfv1 * sin(dphi[tid] + phi0)
                        + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick);
        dphi[tid] += drift_coef * denergy[tid];
    }
}

extern "C"
__global__ void kick_drift_up_turns(double * __restrict__ dphi,
                         double * __restrict__ denergy,
                         double * __restrict__ xp,
                         double * __restrict__ yp,
                         const double * __restrict__ drift_coef,
                         const double * __restrict__ rfv1,
                         const double * __restrict__ rfv2,
                         const double * __restrict__ phi0,
                         const double * __restrict__ phi12,
                         const double hratio,
                         const int nr_particles,
                         const double * __restrict__ acc_kick,
                         int turn,
                         const int nturns,
                         const int dturns,
                         int profile) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < nr_particles)
    {
        double current_dphi = dphi[tid];
        double current_denergy = denergy[tid];

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

extern "C"
__global__ void kick_drift_down_turns(double * __restrict__ dphi,
                         double * __restrict__ denergy,
                         double * __restrict__ xp,
                         double * __restrict__ yp,
                         const double * __restrict__ drift_coef,
                         const double * __restrict__ rfv1,
                         const double * __restrict__ rfv2,
                         const double * __restrict__ phi0,
                         const double * __restrict__ phi12,
                         const double hratio,
                         const int nr_particles,
                         const double * __restrict__ acc_kick,
                         int turn,
                         const int dturns,
                         int profile) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < nr_particles)
    {
        double current_dphi = dphi[tid];
        double current_denergy = denergy[tid];

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