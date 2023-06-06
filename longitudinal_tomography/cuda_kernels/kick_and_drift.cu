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
__global__ void kick_up(const double *dphi,
                        double *denergy,
                        const double rfv1,
                        const double rfv2,
                        const double phi0,
                        const double phi12,
                        const double hratio,
                        const int nr_particles,
                        const double acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    denergy[tid] += rfv1 * sin(dphi[tid] + phi0)
                + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick;
}

extern "C"
__global__ void kick_down(const double *dphi,
                          double *denergy,
                          const double rfv1,
                          const double rfv2,
                          const double phi0,
                          const double phi12,
                          const double hratio,
                          const int nr_particles,
                          const double acc_kick) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    denergy[tid] -= rfv1 * sin(dphi[tid] + phi0)
                + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick;
}

extern "C"
__global__ void drift_up(double *dphi,
                         const double *denergy,
                         const double drift_coef,
                         const int nr_particles) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    dphi[tid] -= drift_coef * denergy[tid];
}

extern "C"
__global__ void drift_down(double *dphi,
                         const double *denergy,
                         const double drift_coef,
                         const int nr_particles) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    dphi[tid] += drift_coef * denergy[tid];
}

extern "C"
__global__ void kick_drift_up_simultaneously(double *dphi,
                         double *denergy,
                         const double drift_coef,
                         const double rfv1,
                         const double rfv2,
                         const double phi0,
                         const double phi12,
                         const double hratio,
                         const int nr_particles,
                         const double acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    dphi[tid] -= drift_coef * denergy[tid];
    denergy[tid] += rfv1 * sin(dphi[tid] + phi0)
                + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick;
}

extern "C"
__global__ void kick_drift_down_simultaneously(double *dphi,
                         double *denergy,
                         const double drift_coef,
                         const double rfv1,
                         const double rfv2,
                         const double phi0,
                         const double phi12,
                         const double hratio,
                         const int nr_particles,
                         const double acc_kick) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    denergy[tid] -= rfv1 * sin(dphi[tid] + phi0)
                    + rfv2 * sin(hratio * (dphi[tid] + phi0 - phi12)) - acc_kick;
    dphi[tid] += drift_coef * denergy[tid];
}