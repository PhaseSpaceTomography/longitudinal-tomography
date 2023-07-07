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
__global__ void back_project(double * __restrict__ weights,                     // inn/out
                             int * __restrict__ flat_points,                    // inn
                             const double * __restrict__ flat_profiles,         // inn
                             const int npart, const int nprof) {                // inn
    const int BLOCK_SIZE = 32;
    const int ITEMS_PER_ARRAY = 16;
    const int ITEMS_PER_IT = BLOCK_SIZE * ITEMS_PER_ARRAY;
    int iterations = (nprof + ITEMS_PER_IT - 1) / ITEMS_PER_IT;

    double aggregate = 0.0;

    for(int i = 0; i < iterations; i++)
    {
        typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;

        // allocate shared memory for BlockReduce
        __shared__ typename BlockReduce::TempStorage temp_storage;

        // int items_per_thread = (nprof + blockDim.x - 1) / blockDim.x;
        double weight_prof[ITEMS_PER_ARRAY];

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

//// Implementation with atomic operations and block_size = npart * nprof
// extern "C"
// __global__ void back_project(double *weights,                     // inn/out
//                              int *flat_points,                    // inn
//                              const double *flat_profiles,         // inn
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
__global__ void project(double * __restrict__ flat_rec,         // inn/out
                        const int * __restrict__ flat_points,   // inn
                        const double * __restrict__ weights,    // inn
                        const int npart, const int nprof) {     // inn
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < npart * nprof)
    {
        int idx = flat_points[tid];
        atomicAdd(&flat_rec[idx], weights[tid / nprof]);
    }
}

// extern "C"
// __global__ void normalize(double *flat_rec, // inn/out
//                const int nprof,
//                const int nbins) {
//     // TODO
//     double sum_waterfall = 0.0;
// #pragma omp parallel for reduction(+ : sum_waterfall)
//     for (int i = 0; i < nprof; i++) {
//         double sum_profile = 0;
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
__global__ void clip(double *array, // inn/out
          const int length,
          const double clip_val) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < length)
    {
        if (array[tid] < clip_val)
            array[tid] = clip_val;
    }
}


extern "C"
__global__ void find_difference_profile(double * __restrict__ diff_prof,    // out
                             const double * __restrict__ flat_rec,          // inn
                             const double * __restrict__ flat_profiles,     // inn
                             const int all_bins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid; i < all_bins; i += blockDim.x * gridDim.x)
        if (i < all_bins)
            diff_prof[i] = flat_profiles[i] - flat_rec[i];
}

// extern "C"
// __global__ double discrepancy(const double *diff_prof,   // inn
//                    const int nprof,
//                    const int nbins) {
//     int all_bins = nprof * nbins;
//     double squared_sum = 0;

//     int tid = threadIdx.x + blockDim.x * blockIdx.x;
//     for (int i = tid; i < all_bins; i += blockDim.x * gridDim.x) {
//         squared_sum += pow(diff_prof[i], 2.0);
//     }

//     return sqrt(squared_sum / (nprof * nbins));
// }

extern "C"
__global__ void compensate_particle_amount(double * __restrict__ diff_prof,     // inn/out
                                const double * __restrict__ rparts,             // inn
                                const int nprof,
                                const int nbins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < nprof * nbins) {
        diff_prof[tid] *= rparts[tid];
    }
}

// // Parallel reduction?
// extern "C"
// __global__ double max_2d(double **arr,  // inn
//               const int x_axis,
//               const int y_axis) {
//     double max_bin_val = 0;
//     for (int i = 0; i < y_axis; i++)
//         for (int j = 0; j < x_axis; j++)
//             if (max_bin_val < arr[i][j])
//                 max_bin_val = arr[i][j];
//     return max_bin_val;
// }

// // Parallel reduction?
// extern "C"
// __global__ double max_1d(double *arr, const int length) {
//     double max_bin_val = 0;
//     for (int i = 0; i < length; i++)
//         if (max_bin_val < arr[i])
//             max_bin_val = arr[i];
//     return max_bin_val;
// }

// Atomic add?
extern "C"
__global__ void count_particles_in_bin(double * __restrict__ rparts,    // out
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
__global__ void calculate_reciprocal(double *rparts,   // inn/out
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

extern "C"
__global__ void create_flat_points(int *flat_points,    // inn/out
                        const int npart,
                        const int nprof,
                        const int nbins) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < npart * nprof)
        flat_points[tid] += nbins * (tid % nprof);
}