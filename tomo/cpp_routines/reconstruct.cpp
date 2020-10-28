#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// Back projection using flattened arrays
extern "C" void back_project(double *  weights,                     // inn/out
                             int ** __restrict__ flat_points,       // inn
                             const double *  flat_profiles,         // inn
                             const int npart, const int nprof){     // inn
#pragma omp parallel for
    for (int i = 0; i < npart; i++)
        for (int j = 0; j < nprof; j++)
            weights[i] += flat_profiles[flat_points[i][j]];
}

// Projections using flattened arrays
extern "C" void project(double *  flat_rec,                     // inn/out
                        int ** __restrict__ flat_points,        // inn
                        const double *  __restrict__ weights,   // inn
                        const int npart, const int nprof){      // inn
    for (int i = 0; i < npart; i++)
        for (int j = 0; j < nprof; j++)
            flat_rec[flat_points[i][j]] += weights[i];
}

void normalize(double * __restrict__ flat_rec, // inn/out
               const int nprof,
               const int nbins){
    int i, j;

double sum_waterfall = 0.0;
#pragma omp parallel for reduction(+ : sum_waterfall)
    for(i=0; i < nprof; i++){
        double sum_profile = 0;
        for(j=0; j < nbins; j++)
            sum_profile += flat_rec[i * nbins + j];
        for(j=0; j < nbins; j++)
            flat_rec[i * nbins + j] /= sum_profile;
        sum_waterfall += sum_profile;
    }

    if(sum_waterfall <= 0)
        throw std::runtime_error("Phase space reduced to zeroes!");
}

void clip(double * __restrict__ array, // inn/out
          const int length,
          const double clip_val){
    bool positive_flag = false;
    int i;

#pragma omp parallel for
    for(i=0; i < length; i++)
        if(array[i] < clip_val)
            array[i] = clip_val;
        else if(!positive_flag)
            positive_flag = true;

    if(!positive_flag)
        throw std::runtime_error("All of phase space got reduced to zeroes");
} 


void find_difference_profile(double * __restrict__ diff_prof,           // out
                             const double * __restrict__ flat_rec,      // inn
                             const double * __restrict__ flat_profiles, // inn
                             const int all_bins){
#pragma omp parallel for
    for(int i = 0; i < all_bins; i++)
        diff_prof[i] = flat_profiles[i] - flat_rec[i];
}

double discrepancy(const double * __restrict__ diff_prof,   // inn
                   const int nprof,
                   const int nbins){
    int all_bins = nprof * nbins;
    double squared_sum = 0;

    for(int i=0; i < all_bins; i++){
        squared_sum += std::pow(diff_prof[i], 2.0);
    }

    return std::sqrt(squared_sum / (nprof * nbins));
}

void compensate_particle_amount(double * __restrict__ diff_prof,        // inn/out
                                double ** __restrict__ rparts,          // inn
                                const int nprof,
                                const int nbins){
    int flat_index = 0, i, j;
    for(i=0; i < nprof; i++)
        for(j=0; j < nbins; j++){
            flat_index = i * nbins + j;
            diff_prof[flat_index] *= rparts[i][j];  
        }
}

double max_2d(double **  __restrict__ arr,  // inn
              const int x_axis,
              const int y_axis){
    double max_bin_val = 0;
    int i, j;
    for(i=0; i < y_axis; i++)
        for(j=0; j < x_axis; j++)
            if(max_bin_val < arr[i][j])
                max_bin_val = arr[i][j];
    return max_bin_val;
}

void count_particles_in_bin(double ** __restrict__ rparts,      // out
                            const int ** __restrict__ xp,       // inn
                            const int nprof,
                            const int npart){
    int index, i, j;
    for(i=0; i < npart; i++)
        for(j=0; j < nprof; j++){
            index = xp[i][j];
            rparts[j][index] += 1;
        }
}

void reciprocal_particles(double **  __restrict__ rparts,   // out
                          const int ** __restrict__ xp,     // inn
                          const int nbins,
                          const int nprof,
                          const int npart){
    int i, j;

    // initiating rparts to 0
#pragma omp parallel for
    for(i=0; i < nprof; i++)
        for(j=0; j < nbins; j++)
            rparts[i][j] = 0.0;

    count_particles_in_bin(rparts, xp, nprof, npart);

    int max_bin_val = max_2d(rparts, nbins, nprof);

    // Setting 0's to 1's to avoid zero division
#pragma omp parallel for
    for(i = 0; i < nprof; i++)
        for(j = 0; j < nbins; j++)
            if(rparts[i][j] == 0.0)
                rparts[i][j] = 1.0;

    // Creating reciprocal
#pragma omp parallel for
    for(i = 0; i < nprof; i++)
        for(j = 0; j < nbins; j++)
                rparts[i][j] = (double) max_bin_val / rparts[i][j];
}

void create_flat_points(const int ** __restrict__ xp,       //inn
                        int ** __restrict__ flat_points,    //out
                        const int npart,
                        const int nprof,
                        const int nbins){
    int i, j;
    // Initiating to the value of xp
    for(i = 0; i < npart; i++)
        for(j = 0; j < nprof; j++)
            flat_points[i][j] = xp[i][j];

    for(i = 0; i < npart; i++)
        for(j = 0; j < nprof; j++)
            flat_points[i][j] += nbins * j;
}

// VERSION 0
// Projections using flattened arrays
// Working original version
// Well tested (12.02.2020)
// < to be removed when new version is prooven to be working correctly >
extern "C" void old_reconstruct(double * __restrict__ weights,            // out
                                const int ** __restrict__ xp,             // inn
                                const double * __restrict__ flat_profiles,// inn
                                double * __restrict__ discr,              // out
                                const int niter,
                                const int nbins,
                                const int npart,
                                const int nprof,
                                const bool verbose){
    int i;

    // Creating arrays...
    int all_bins = nprof * nbins;
    double * flat_rec =  new double[all_bins];
    for(i = 0; i < all_bins; i++)
        flat_rec[i] = 0;

    double * diff_prof =  new double[all_bins];
    
    double** rparts = new double*[nprof];
    for(i = 0; i < nprof; i++)
        rparts[i] = new double[nbins];

    int** flat_points = new int*[npart];
    for(i = 0; i < npart; i++)
        flat_points[i] = new int[nprof];


    // Actual functionality

    reciprocal_particles(rparts, xp, nbins, nprof, npart);

    create_flat_points(xp, flat_points, npart, nprof, nbins);

    back_project(weights, flat_points, flat_profiles, npart, nprof);
    clip(weights, npart, 0.0);
    
    if (verbose)
        std::cout << " Iterating..." << std::endl;
    
    for(int iteration = 0; iteration < niter; iteration++){
        if (verbose)
            std::cout << std::setw(3) << iteration + 1 << std::endl;

        project(flat_rec, flat_points, weights, npart, nprof);
        normalize(flat_rec, nprof, nbins);

        find_difference_profile(diff_prof, flat_rec, flat_profiles, all_bins);

        discr[iteration] = discrepancy(diff_prof, nprof, nbins);

        compensate_particle_amount(diff_prof, rparts, nprof, nbins);

        back_project(weights, flat_points, diff_prof, npart, nprof);
        clip(weights, npart, 0.0);
    } //end for

    // Calculating final discrepancy
    project(flat_rec, flat_points, weights, npart, nprof);
    normalize(flat_rec, nprof, nbins);
    
    find_difference_profile(diff_prof, flat_rec, flat_profiles, all_bins);
    discr[niter] = discrepancy(diff_prof, nprof, nbins);

    // Cleaning
    for(i = 0; i < nprof; i++) {
        delete[] rparts[i];
    }
    for(i = 0; i < npart; i++) {
        delete[] flat_points[i];
    }
    delete[] rparts;
    delete[] flat_points;
    delete[] flat_rec;
    delete[] diff_prof;

    if (verbose)
        std::cout << " Done!" << std::endl;
}

// VERSION 1
// Projections using flattened arrays
// Version returning final recreated phase space
// Not well tested (12.02.2020)
extern "C" void reconstruct(double * __restrict__ weights,             // out
                            const int ** __restrict__ xp,              // inn
                            const double * __restrict__ flat_profiles, // inn
                            double * __restrict__ flat_rec,            // Out
                            double * __restrict__ discr,               // out
                            const int niter,
                            const int nbins,
                            const int npart,
                            const int nprof,
                            const bool verbose){
    int i;

    // Creating arrays...
    int all_bins = nprof * nbins;
    double * diff_prof =  new double[all_bins];
    
    double** rparts = new double*[nprof];
    for(i = 0; i < nprof; i++)
        rparts[i] = new double[nbins];

    int** flat_points = new int*[npart];
    for(i = 0; i < npart; i++)
        flat_points[i] = new int[nprof];


    // Actual functionality

    reciprocal_particles(rparts, xp, nbins, nprof, npart);

    create_flat_points(xp, flat_points, npart, nprof, nbins);

    back_project(weights, flat_points, flat_profiles, npart, nprof);
    clip(weights, npart, 0.0);
    
    if (verbose)
        std::cout << " Iterating..." << std::endl;
    
    for(int iteration = 0; iteration < niter; iteration++){
        if (verbose)
            std::cout << std::setw(3) << iteration + 1 << std::endl;

        project(flat_rec, flat_points, weights, npart, nprof);
        normalize(flat_rec, nprof, nbins);

        find_difference_profile(diff_prof, flat_rec, flat_profiles, all_bins);

        discr[iteration] = discrepancy(diff_prof, nprof, nbins);

        compensate_particle_amount(diff_prof, rparts, nprof, nbins);

        back_project(weights, flat_points, diff_prof, npart, nprof);
        clip(weights, npart, 0.0);
    } //end for

    // Calculating final discrepancy
    project(flat_rec, flat_points, weights, npart, nprof);
    normalize(flat_rec, nprof, nbins);
    
    find_difference_profile(diff_prof, flat_rec, flat_profiles, all_bins);
    discr[niter] = discrepancy(diff_prof, nprof, nbins);

    // Cleaning
    for(i = 0; i < nprof; i++) {
        delete[] rparts[i];
    }
    for(i = 0; i < npart; i++) {
        delete[] flat_points[i];
    }
    delete[] rparts;
    delete[] flat_points;
    delete[] diff_prof;

    if (verbose)
        std::cout << " Done!" << std::endl;
}
