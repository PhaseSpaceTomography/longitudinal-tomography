/**
 * @file reconstruct.cu
 *
 * @author Bernardo Abreu Figueiredo
 * Contact: bernardo.abreu.figueiredo@cern.ch
 *
 * CUDA kernels that handle phase space reconstruction.
 */

#include <cub/block/block_reduce.cuh>

// Back projection using flattened arrays
// implementation with fixed block_size and items_per_array, but variable number of profiles for the reduction
// Must be called with block size 32
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

        // int items_per_thread = (nprof + blockDim.x - 1) / blockDim.x;
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

//// Implementation with atomic operations and block_size = npart * nprof
// extern "C"
// __global__ void back_project(float *weights,                     // inn/out
//                              int *flat_points,                    // inn
//                              const float *flat_profiles,         // inn
//                              const int npart, const int nprof) {  // inn
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;

//     if(tid < npart * nprof)
//     {
//         int idx = flat_points[tid];
//         atomicAdd(&weights[tid / nprof], flat_profiles[idx]);
//     }
// }


// Projections using flattened arrays
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

// extern "C"
// __global__ void normalize(float *flat_rec, // inn/out
//                const int nprof,
//                const int nbins) {
//     // TODO
//     float sum_waterfall = 0.0;
// #pragma omp parallel for reduction(+ : sum_waterfall)
//     for (int i = 0; i < nprof; i++) {
//         float sum_profile = 0;
//         for (int j = 0; j < nbins; j++)
//             sum_profile += flat_rec[i * nbins + j];
//         for (int j = 0; j < nbins; j++)
//             flat_rec[i * nbins + j] /= sum_profile;
//         sum_waterfall += sum_profile;
//     }

//     if (sum_waterfall <= 0)
//         throw std::runtime_error("Phase space reduced to zeroes!");
// }

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

// extern "C"
// __global__ float discrepancy(const float *diff_prof,   // inn
//                    const int nprof,
//                    const int nbins) {
//     int all_bins = nprof * nbins;
//     float squared_sum = 0;

//     int tid = threadIdx.x + blockDim.x * blockIdx.x;
//     for (int i = tid; i < all_bins; i += blockDim.x * gridDim.x) {
//         squared_sum += pow(diff_prof[i], 2.0);
//     }

//     return sqrt(squared_sum / (nprof * nbins));
// }

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

// // Parallel reduction?
// extern "C"
// __global__ float max_2d(float **arr,  // inn
//               const int x_axis,
//               const int y_axis) {
//     float max_bin_val = 0;
//     for (int i = 0; i < y_axis; i++)
//         for (int j = 0; j < x_axis; j++)
//             if (max_bin_val < arr[i][j])
//                 max_bin_val = arr[i][j];
//     return max_bin_val;
// }

// // Parallel reduction?
// extern "C"
// __global__ float max_1d(float *arr, const int length) {
//     float max_bin_val = 0;
//     for (int i = 0; i < length; i++)
//         if (max_bin_val < arr[i])
//             max_bin_val = arr[i];
//     return max_bin_val;
// }

// Atomic add?
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

extern "C"
__global__ void create_flat_points(int *flat_points,    // inn/out
                        const int npart,
                        const int nprof,
                        const int nbins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < npart * nprof)
        flat_points[tid] += nbins * (tid % nprof);
}