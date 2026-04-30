/**
 * @file reconstruct.cu
 *
 * @author Bernardo Abreu Figueiredo, Balazs Paszkal Halmos
 * Contact: bernardo.abreu.figueiredo@cern.ch, balazs.paszkal.halmos@cern.ch
 *
 * CUDA kernels that handle phase space reconstruction for real_int_t precision floating-point numbers.
 */

#include <cub/block/block_reduce.cuh>

#ifdef USE32BITS
    typedef int32_t int_t;
    #ifdef USEINT
        typedef int32_t real_int_t; 
    #else
        typedef float real_int_t;
    #endif
#else
    typedef int64_t int_t;
    #ifdef USEINT
        typedef int64_t real_int_t;
    #else
        typedef double real_int_t;
    #endif
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// Back projection using flattened arrays and a block-wide reduction.
// Implementation with fixed block_size and items_per_array, but variable number of profiles for the reduction
// Must be called with block size it was compiled with (BLOCK_SIZE variable)
extern "C"
__global__ void back_project(real_int_t * __restrict__ weights,                 // inn/out
                             int_t * __restrict__ flat_points,                // inn
                             const real_int_t * __restrict__ flat_profiles,     // inn
                             const int_t npart, const int_t nprof) {            // inn
    const int_t ITEMS_PER_ARRAY = 512 / BLOCK_SIZE;
    const int_t ITEMS_PER_IT = BLOCK_SIZE * ITEMS_PER_ARRAY;
    int_t iterations = (nprof + ITEMS_PER_IT - 1) / ITEMS_PER_IT;

    real_int_t aggregate = 0.0;

    for(int_t i = 0; i < iterations; i++)
    {
        typedef cub::BlockReduce<real_int_t, BLOCK_SIZE> BlockReduce;

        // allocate shared memory for BlockReduce
        __shared__ typename BlockReduce::TempStorage temp_storage;

        real_int_t weight_prof[ITEMS_PER_ARRAY];

        for (int_t j = 0; j < ITEMS_PER_ARRAY; j++)
        {
            int_t index = i * ITEMS_PER_IT + j * blockDim.x + threadIdx.x;
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

// Projection using flattened arrays.
extern "C"
__global__ void project(real_int_t * __restrict__ flat_rec,         // inn/out
                        const int_t * __restrict__ flat_points,   // inn
                        const real_int_t * __restrict__ weights,    // inn
                        const int_t npart, const int_t nprof) {     // inn
    int_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    int_t stride = blockDim.x * gridDim.x;

    for (int_t i = tid; i < npart * nprof; i += stride)
    {
        int_t idx = flat_points[i];
        #ifdef USEINT
            atomicAdd(
                reinterpret_cast<unsigned long long*>(&flat_rec[idx]),
                static_cast<unsigned long long>(weights[i / nprof])
            );
        #else
            atomicAdd(&flat_rec[idx], weights[i / nprof]);
        #endif
    }
}

// Array clipping function to set values below a threshold
// to the respective value.
extern "C"
__global__ void clip(real_int_t *array,             // inn/out
                     const int_t length,
                     const real_int_t clip_val) {
    int_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    int_t stride = blockDim.x * gridDim.x;
    
    for (int_t i = tid; i < length; i += stride)
    {
        if (array[i] < clip_val)
            array[i] = clip_val;
    }
}

// Calculates the difference between the reconstructed profile
// and the flat profiles.
// This function iterates, however to reduce multiple iterations,
// the amount of threads should be at least equal to all_bins if possible.
extern "C"
__global__ void find_difference_profile(real_int_t * __restrict__ diff_prof,            // out
                                        const real_int_t * __restrict__ flat_rec,       // inn
                                        const real_int_t * __restrict__ flat_profiles,  // inn
                                        const int_t all_bins) {
    int_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int_t i = tid; i < all_bins; i += blockDim.x * gridDim.x)
        if (i < all_bins)
            diff_prof[i] = flat_profiles[i] - flat_rec[i];
}

// Multiplies the profile differences with the reciprocal particle array
// to compensate for the amount of particles.
extern "C"
__global__ void compensate_particle_amount(real_int_t * __restrict__ diff_prof,     // inn/out
                                           const real_int_t * __restrict__ rparts,  // inn
                                           const int_t nprof,
                                           const int_t nbins) {
    int_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    int_t stride = blockDim.x * gridDim.x;

    for (int_t i = tid; i < nprof * nbins; i += stride) {
        diff_prof[i] *= rparts[i];
    }
}

// Counts the particles in each bin.
extern "C"
__global__ void count_particles_in_bin(real_int_t * __restrict__ rparts,    // out
                                       const int_t * __restrict__ xp,     // inn
                                       const int_t nprof,
                                       const int_t npart,
                                       const int_t nbins) {
    int_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    int_t stride = blockDim.x * gridDim.x;
    
    for (int_t i = tid; i < npart * nprof; i += stride)
    {
        int_t j = i % nprof;
        int_t bin = xp[i];
        #ifdef USEINT
            atomicAdd(reinterpret_cast<unsigned long long*>(&rparts[j * nbins + bin]), 1);
        #else
            atomicAdd(&rparts[j * nbins + bin], 1);
        #endif
    }
}

// Calculates the reciprocal of the counted particles per bin.
extern "C"
__global__ void calculate_reciprocal(real_int_t *rparts,        // inn/out
                                     const int_t nbins,
                                     const int_t nprof,
                                     const real_int_t maxVal) {
    const int_t all_bins = nprof * nbins;

    // Setting 0's to 1's to avoid zero division and creating reciprocal
    int_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    int_t stride = blockDim.x * gridDim.x;
    
    for (int_t i = tid; i < all_bins; i += stride) {
        if (rparts[i] == static_cast<real_int_t>(0.0))
            rparts[i] = 1.0;
        rparts[i] = maxVal / rparts[i];
    }
}

// Creates a flattened representation of the particle coordinates
// used for indexing. 
extern "C"
__global__ void create_flat_points(int_t *flat_points,    // inn/out
                                   const int_t npart,
                                   const int_t nprof,
                                   const int_t nbins) {
    int_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    int_t stride = blockDim.x * gridDim.x;
    
    for (int_t i = tid; i < npart * nprof; i += stride)
        flat_points[i] += nbins * (i % nprof);
}