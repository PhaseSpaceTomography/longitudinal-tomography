/**
 * @file reconstruct.cu
 *
 * @author Bernardo Abreu Figueiredo
 * Contact: bernardo.abreu.figueiredo@cern.ch
 *
 * CUDA kernels that handle phase space reconstruction for single precision floating-point numbers.
 */

#include <cub/block/block_reduce.cuh>

// Back projection using flattened arrays and a block-wide reduction.
// Implementation with fixed block_size and items_per_array, but variable number of profiles for the reduction
// Must be called with block size 32.
extern "C"
__global__ void back_project(float * __restrict__ weights,                     // inn/out
                             int * __restrict__ flat_points,                    // inn
                             const float * __restrict__ flat_profiles,         // inn
                             const int npart, const int nprof) {                // inn
    const int BLOCK_SIZE = 32;
    const int ITEMS_PER_ARRAY = 16;
    const int ITEMS_PER_IT = BLOCK_SIZE * ITEMS_PER_ARRAY;
    int iterations = (nprof + ITEMS_PER_IT - 1) / ITEMS_PER_IT;

    float aggregate = 0.0f;

    for(int i = 0; i < iterations; i++)
    {
        typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;

        // allocate shared memory for BlockReduce
        __shared__ typename BlockReduce::TempStorage temp_storage;

        float weight_prof[ITEMS_PER_ARRAY];

        for (int j = 0; j < ITEMS_PER_ARRAY; j++)
        {
            int index = i * ITEMS_PER_IT + j * blockDim.x + threadIdx.x;
            if (index < nprof)
                weight_prof[j] = flat_profiles[flat_points[blockIdx.x * nprof + index]];
            else
                weight_prof[j] = 0.0f;
        }

        __syncthreads();

        aggregate += BlockReduce(temp_storage).Sum(weight_prof);
    }

    if (threadIdx.x == 0)
        weights[blockIdx.x] += aggregate;
}

// Projection using flattened arrays.
// This function does not iterate, so the
// amount of threads should be at least equal to the product of npart and nprof.
extern "C"
__global__ void project(float * __restrict__ flat_rec,         // inn/out
                        const int * __restrict__ flat_points,   // inn
                        const float * __restrict__ weights,    // inn
                        const int npart, const int nprof) {     // inn
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < npart * nprof)
    {
        int idx = flat_points[tid];
        atomicAdd(&flat_rec[idx], weights[tid / nprof]);
    }
}

// Array clipping function to set values below a threshold
// to the respective value.
// This function does not iterate, so the
// amount of threads should be at least equal to the length.
extern "C"
__global__ void clip(float *array, // inn/out
          const int length,
          const double clip_val) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < length)
    {
        if (array[tid] < (float) clip_val)
            array[tid] = (float) clip_val;
    }
}

// Calculates the difference between the reconstructed profile
// and the flat profiles.
// This function iterates, however to reduce multiple iterations,
// the amount of threads should be at least equal to all_bins if possible.
extern "C"
__global__ void find_difference_profile(float * __restrict__ diff_prof,    // out
                             const float * __restrict__ flat_rec,          // inn
                             const float * __restrict__ flat_profiles,     // inn
                             const int all_bins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid; i < all_bins; i += blockDim.x * gridDim.x)
        if (i < all_bins)
            diff_prof[i] = flat_profiles[i] - flat_rec[i];
}

// Multiplies the profile differences with the reciprocal particle array
// to compensate for the amount of particles.
// This function does not iterate, so the
// amount of threads should be at least equal to the product of nprof and nbins.
extern "C"
__global__ void compensate_particle_amount(float * __restrict__ diff_prof,     // inn/out
                                const float * __restrict__ rparts,             // inn
                                const int nprof,
                                const int nbins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nprof * nbins) {
        diff_prof[tid] *= rparts[tid];
    }
}

// Counts the particles in each bin.
// This function does not iterate, so the
// amount of threads should be at least equal to the product of npart and nprof.
extern "C"
__global__ void count_particles_in_bin(float * __restrict__ rparts,    // out
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

// Calculates the reciprocal of the counted particles per bin.
// This function does not iterate, so the
// amount of threads should be at least equal to the product of nprof and nbins.
extern "C"
__global__ void calculate_reciprocal(float *rparts,   // inn/out
                          const int nbins,
                          const int nprof,
                          const double maxVal) {
    const int all_bins = nprof * nbins;

    // Setting 0's to 1's to avoid zero division and creating reciprocal
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < all_bins) {
        if (rparts[tid] == 0.0f)
            rparts[tid] = 1.0f;
        rparts[tid] = (float) maxVal / rparts[tid];
    }
}

// Creates a flattened representation of the particle coordinates
// used for indexing. 
// This function does not iterate, so the
// amount of threads should be at least equal to the product of npart and nprof.
extern "C"
__global__ void create_flat_points(int *flat_points,    // inn/out
                        const int npart,
                        const int nprof,
                        const int nbins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < npart * nprof)
        flat_points[tid] += nbins * (tid % nprof);
}