/**
 * @file reconstruct.cu
 *
 * @author Bernardo Abreu Figueiredo
 * Contact: bernardo.abreu.figueiredo@cern.ch
 *
 * CUDA kernels that handle phase space reconstruction for T precision floating-point numbers.
 */

#include <cub/block/block_reduce.cuh>

// Back projection using flattened arrays and a block-wide reduction.
// Implementation with fixed block_size and items_per_array, but variable number of profiles for the reduction
// Must be called with block size 32.
template <typename T>
__device__ void back_project(T * __restrict__ weights,                     // inn/out
                             int * __restrict__ flat_points,               // inn
                             const T * __restrict__ flat_profiles,         // inn
                             const int npart, const int nprof) {           // inn
    const int BLOCK_SIZE = 32;
    const int ITEMS_PER_ARRAY = 16;
    const int ITEMS_PER_IT = BLOCK_SIZE * ITEMS_PER_ARRAY;
    int iterations = (nprof + ITEMS_PER_IT - 1) / ITEMS_PER_IT;

    T aggregate = 0.0;

    for(int i = 0; i < iterations; i++)
    {
        typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;

        // allocate shared memory for BlockReduce
        __shared__ typename BlockReduce::TempStorage temp_storage;

        T weight_prof[ITEMS_PER_ARRAY];

        for (int j = 0; j < ITEMS_PER_ARRAY; j++)
        {
            int index = i * ITEMS_PER_IT + j * blockDim.x + threadIdx.x;
            if (index < nprof)
                weight_prof[j] = flat_profiles[flat_points[blockIdx.x * nprof + index]];
            else
                weight_prof[j] = 0.0;
        }

        __syncthreads();

        aggregate += BlockReduce(temp_storage).Sum(weight_prof);
    }

    if (threadIdx.x == 0)
        weights[blockIdx.x] += aggregate;
}

extern "C" __global__ void back_project_double(double * __restrict__ weights,
                             int * __restrict__ flat_points,
                             const double * __restrict__ flat_profiles,
                             const int npart, const int nprof) {
    back_project<double>(weights, flat_points, flat_profiles, npart, nprof);
}

extern "C" __global__ void back_project_float(float * __restrict__ weights,
                             int * __restrict__ flat_points,
                             const float * __restrict__ flat_profiles,
                             const int npart, const int nprof) {
    back_project<float>(weights, flat_points, flat_profiles, npart, nprof);
}

// Projection using flattened arrays.
// This function does not iterate, so the
// amount of threads should be at least equal to the product of npart and nprof.
template <typename T>
__device__ void project(T * __restrict__ flat_rec,              // inn/out
                        const int * __restrict__ flat_points,   // inn
                        const T * __restrict__ weights,         // inn
                        const int npart, const int nprof) {     // inn
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < npart * nprof)
    {
        int idx = flat_points[tid];
        atomicAdd(&flat_rec[idx], weights[tid / nprof]);
    }
}

extern "C" __global__ void project_double(double * __restrict__ flat_rec,
                        const int * __restrict__ flat_points,
                        const double * __restrict__ weights,
                        const int npart, const int nprof) {
    project<double>(flat_rec, flat_points, weights, npart, nprof);
}

extern "C" __global__ void project_float(float * __restrict__ flat_rec,
                        const int * __restrict__ flat_points,
                        const float * __restrict__ weights,
                        const int npart, const int nprof) {
    project<float>(flat_rec, flat_points, weights, npart, nprof);
}

// Array clipping function to set values below a threshold
// to the respective value.
// This function does not iterate, so the
// amount of threads should be at least equal to the length.
template <typename T>
__device__ void clip(T *array, // inn/out
          const int length,
          const double clip_val) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < length)
    {
        if (array[tid] < clip_val)
            array[tid] = clip_val;
    }
}

extern "C" __global__ void clip_double(double *array,
                    const int length,
                    const double clip_val) {
    clip<double>(array, length, clip_val);
}

extern "C" __global__ void clip_float(float *array,
                    const int length,
                    const double clip_val) {
    clip<float>(array, length, clip_val);
}

// Calculates the difference between the reconstructed profile
// and the flat profiles.
// This function iterates, however to reduce multiple iterations,
// the amount of threads should be at least equal to all_bins if possible.
template <typename T>
__device__ void find_difference_profile(T * __restrict__ diff_prof,    // out
                             const T * __restrict__ flat_rec,          // inn
                             const T * __restrict__ flat_profiles,     // inn
                             const int all_bins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid; i < all_bins; i += blockDim.x * gridDim.x)
        if (i < all_bins)
            diff_prof[i] = flat_profiles[i] - flat_rec[i];
}

extern "C" __global__ void find_difference_profile_double(double * __restrict__ diff_prof,
                            const double * __restrict__ flat_rec,
                            const double * __restrict__ flat_profiles,
                            const int all_bins) {
    find_difference_profile<double>(diff_prof, flat_rec, flat_profiles, all_bins);
}

extern "C" __global__ void find_difference_profile_float(float * __restrict__ diff_prof,
                            const float * __restrict__ flat_rec,
                            const float * __restrict__ flat_profiles,
                            const int all_bins) {
    find_difference_profile<float>(diff_prof, flat_rec, flat_profiles, all_bins);
}

// Multiplies the profile differences with the reciprocal particle array
// to compensate for the amount of particles.
// This function does not iterate, so the
// amount of threads should be at least equal to the product of nprof and nbins.
template <typename T>
__device__ void compensate_particle_amount(T * __restrict__ diff_prof,     // inn/out
                                const T * __restrict__ rparts,             // inn
                                const int nprof,
                                const int nbins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nprof * nbins) {
        diff_prof[tid] *= rparts[tid];
    }
}

extern "C" __global__ void compensate_particle_amount_double(double * __restrict__ diff_prof,     // inn/out
                                const double * __restrict__ rparts,             // inn
                                const int nprof,
                                const int nbins) {
    compensate_particle_amount<double>(diff_prof, rparts, nprof, nbins);
}

extern "C" __global__ void compensate_particle_amount_float(float * __restrict__ diff_prof,     // inn/out
                                const float * __restrict__ rparts,             // inn
                                const int nprof,
                                const int nbins) {
    compensate_particle_amount<float>(diff_prof, rparts, nprof, nbins);
}

// Counts the particles in each bin.
// This function does not iterate, so the
// amount of threads should be at least equal to the product of npart and nprof.
template <typename T>
__device__ void count_particles_in_bin(T * __restrict__ rparts,    // out
                            const int * __restrict__ xp,                // inn
                            const int nprof,
                            const int npart,
                            const int nbins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < npart * nprof)
    {
        int j = tid % nprof;
        int bin = xp[tid];
        atomicAdd(&rparts[j * nbins + bin], 1);
    }
}

extern "C" __global__ void count_particles_in_bin_double(double * __restrict__ rparts,
                            const int * __restrict__ xp,
                            const int nprof,
                            const int npart,
                            const int nbins) {
    count_particles_in_bin<double>(rparts, xp, nprof, npart, nbins);
}

extern "C" __global__ void count_particles_in_bin_float(float * __restrict__ rparts,
                            const int * __restrict__ xp,
                            const int nprof,
                            const int npart,
                            const int nbins) {
    count_particles_in_bin<float>(rparts, xp, nprof, npart, nbins);
}

// Calculates the reciprocal of the counted particles per bin.
// This function does not iterate, so the
// amount of threads should be at least equal to the product of nprof and nbins.
template <typename T>
__device__ void calculate_reciprocal(T *rparts,   // inn/out
                          const int nbins,
                          const int nprof,
                          const double maxVal) {
    const int all_bins = nprof * nbins;

    // Setting 0's to 1's to avoid zero division and creating reciprocal
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < all_bins) {
        if (rparts[tid] == 0.0)
            rparts[tid] = 1.0;
        rparts[tid] = maxVal / rparts[tid];
    }
}

extern "C" __global__ void calculate_reciprocal_double(double *rparts,
                          const int nbins,
                          const int nprof,
                          const double maxVal) {
    calculate_reciprocal<double>(rparts, nbins, nprof, maxVal);
}

extern "C" __global__ void calculate_reciprocal_float(float *rparts,
                          const int nbins,
                          const int nprof,
                          const double maxVal) {
    calculate_reciprocal<float>(rparts, nbins, nprof, maxVal);
}

// Creates a flattened representation of the particle coordinates
// used for indexing. 
// This function does not iterate, so the
// amount of threads should be at least equal to the product of npart and nprof.
extern "C" __global__ void create_flat_points(int *flat_points,    // inn/out
                        const int npart,
                        const int nprof,
                        const int nbins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < npart * nprof)
        flat_points[tid] += nbins * (tid % nprof);
}