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


void project_not_flat(double **  rec,                           // inn/out
                      const int ** __restrict__ xp,             // inn
                      const double *  __restrict__ weights,     // inn
                      const int npart, const int nprof){        // inn
    for (int i = 0; i < npart; i++)
        for (int j = 0; j < nprof; j++)
            rec[j][xp[i][j]] += weights[i];
}

void back_project_not_flat(double *  weights,                     // inn/out
                           const int ** __restrict__ xp,          // inn
                           const double **  profiles,             // inn
                           const int nbins,
                           const int npart,
                           const int nprof){
    double sum_particle;

/*#pragma acc kernels loop copyin(profiles[:nprof][:nbins])\
                         copyout(weights[:npart])\
                         private(sum_particle)\
                         device_type(nvidia) vector_length(32)*/
#pragma omp parallel for
    for (int i = 0; i < npart; i++){
        sum_particle = 0;
        for (int j = 0; j < nprof; j++)
            sum_particle += profiles[j][xp[i][j]];
        weights[i] += sum_particle;
    }
}

void back_project_not_flat(double *  weights,               // inn/out
                           const int ** __restrict__ xp,    // inn
                           double **  profiles,             // inn
                           const int nbins,
                           const int npart,
                           const int nprof){
    double sum_particle;

#pragma omp parallel for
    for (int i = 0; i < npart; i++){
        sum_particle = 0;
        for (int j = 0; j < nprof; j++)
            sum_particle += profiles[j][xp[i][j]];
        weights[i] += sum_particle;
    }
}

void suppress_zeros_norm(double * __restrict__ flat_rec, // inn/out
                        const int nprof,
                        const int nbins){
    int i, j;
    bool positive_flag = false;
    
    // surpressing zeros
    for(i=0; i < nprof * nbins; i++)
        if(flat_rec[i] < 0.0)
            flat_rec[i] = 0.0;
        else if(!positive_flag)
            positive_flag = true;

    if(!positive_flag)
        throw std::runtime_error("All of phase space got reduced to zeroes");
      
    for(i=0; i < nprof; i++){
        double sum_profile = 0;
        for(j=0; j < nbins; j++)
            sum_profile += flat_rec[i * nbins + j];
        for(j=0; j < nbins; j++)
            flat_rec[i * nbins + j] /= sum_profile;
    }
}

void suppress_zeros_norm(double ** __restrict__ rec, // inn/out
                         const int nprof,
                         const int nbins){
    bool positive_flag = false;
    for(int i=0; i < nprof; i++){
        for(int j=0; j < nbins; j++){
            if(rec[i][j] < 0.0)
                rec[i][j] = 0.0; 
            else if (!positive_flag)
                positive_flag = true;
        }
    }

    if(!positive_flag)
        throw std::runtime_error("All of phase space got reduced to zeroes");

    for(int i=0; i < nprof; i++){
        double sum_profile = 0;
        for(int j=0; j < nbins; j++)
            sum_profile += rec[i][j];
        for(int j=0; j < nbins; j++)
            rec[i][j] /= sum_profile;
    }
}

void find_difference_profile(double * __restrict__ diff_prof,           // out
                             const double * __restrict__ flat_rec,      // inn
                             const double * __restrict__ flat_profiles, // inn
                             const int all_bins){
    for(int i = 0; i < all_bins; i++)
        diff_prof[i] = flat_profiles[i] - flat_rec[i];
}

void find_difference_profile(double ** __restrict__ diff_prof,           // out
                             double ** __restrict__ rec,      // inn
                             const double ** __restrict__ profiles, // inn
                             const int nprof,
                             const int nbins){
    for(int i = 0; i < nprof; i++)
        for(int j = 0; j < nbins; j++)
            diff_prof[i][j] = profiles[i][j] - rec[i][j];
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

double discrepancy(double ** __restrict__ diff_prof,  // inn
                   const int nprof,
                   const int nbins){
    double squared_sum = 0;
    for(int i=0; i < nprof; i++)
        for(int j = 0; j < nbins; j++)
            squared_sum += std::pow(diff_prof[i][j], 2.0);

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

void compensate_particle_amount(double ** __restrict__ diff_prof,       // inn/out
                                double ** __restrict__ rparts,          // inn
                                const int nprof,
                                const int nbins){
    int flat_index = 0, i, j;
    for(i=0; i < nprof; i++)
        for(j=0; j < nbins; j++){
            diff_prof[i][j] *= rparts[i][j];  
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
    for(i=0; i < nprof; i++)
        for(j=0; j < nbins; j++)
            rparts[i][j] = 0.0;

    count_particles_in_bin(rparts, xp, nprof, npart);

    int max_bin_val = max_2d(rparts, nbins, nprof);

    // Setting 0's to 1's to avoid zero division
    for(i = 0; i < nprof; i++)
        for(j = 0; j < nbins; j++)
            if(rparts[i][j] == 0.0)
                rparts[i][j] = 1.0;

    // Creating reciprocal
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

// VERSION 1
// Here i will try with non-flat arrays
extern "C" void reconstruct(double * __restrict__ weights,              // out
                            const int ** __restrict__ xp,               // inn
                            const double ** __restrict__ profiles,      // inn
                            const int niter,
                            const int nbins,
                            const int npart,
                            const int nprof){
    int i;

    // Creating arrays...
    int all_bins = nprof * nbins;

    double * discr = new double[niter + 1];
    for(i=0; i < niter + 1; i++)
        discr[i] = 0;

    double** diff_prof = new double*[nprof];    
    double** rec = new double*[nprof];
    double** rparts = new double*[nprof];
    for(i = 0; i < nprof; i++){
        diff_prof[i] = new double[nbins];
        rparts[i] = new double[nbins];
        rec[i] = new double[nbins];
    }

    for (int i = 0; i < nprof; i++)
        for (int j = 0; j < nbins; j++)
            rec[i][j] = 0;

    // Actual functionality

    reciprocal_particles(rparts, xp, nbins, nprof, npart);

    back_project_not_flat(weights, xp, profiles, nbins, npart, nprof);

    for(int iteration = 0; iteration < niter; iteration++){
        std::cout << "Iteration: " << iteration + 1 << " of " << niter << std::endl;

        project_not_flat(rec, xp, weights, npart, nprof);
        
        suppress_zeros_norm(rec, nprof, nbins);

        find_difference_profile(diff_prof, rec, profiles, nprof, nbins);

        discr[iteration] = discrepancy(diff_prof, nprof, nbins);

        compensate_particle_amount(diff_prof, rparts, nprof, nbins);


        back_project_not_flat(weights, xp, diff_prof, nbins, npart, nprof);
    }

    // Calculating final discrepancy
    project_not_flat(rec, xp, weights, npart, nprof);
    suppress_zeros_norm(rec, nprof, nbins);
    
    find_difference_profile(diff_prof, rec, profiles, nprof, nbins);
    discr[niter] = discrepancy(diff_prof, nprof, nbins);

    // std::cout << "Final difference: " << discr[niter] << std::endl;

    // Cleaning
    for(i = 0; i < nprof; i++) {
        delete[] rparts[i];
        delete[] rec[i];
        delete[] diff_prof[i];
    }
    delete[] rparts, rec, discr, diff_prof;
}


// VERSION 0
// Projections using flattened arrays
// Working original version
extern "C" void reconstruct_v0(double * __restrict__ weights,              // out
                            const int ** __restrict__ xp,               // inn
                            const double * __restrict__ flat_profiles,  // inn
                            const int niter,
                            const int nbins,
                            const int npart,
                            const int nprof){
    int i;

    // Creating arrays...
    int all_bins = nprof * nbins;
    double * flat_rec =  new double[all_bins];
    for(i = 0; i < all_bins; i++)
        flat_rec[i] = 0;

    double * diff_prof =  new double[all_bins];

    double * discr = new double[niter + 1];
    for(i=0; i < niter + 1; i++)
        discr[i] = 0;
    
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

    for(int iteration = 0; iteration < niter; iteration++){
        std::cout << "Iteration: " << iteration + 1 << " of " << niter << std::endl;

        project(flat_rec, flat_points, weights, npart, nprof);
        suppress_zeros_norm(flat_rec, nprof, nbins);

        find_difference_profile(diff_prof, flat_rec, flat_profiles, all_bins);

        discr[iteration] = discrepancy(diff_prof, nprof, nbins);

        compensate_particle_amount(diff_prof, rparts, nprof, nbins);

        back_project(weights, flat_points, diff_prof, npart, nprof);
    }

    // Calculating final discrepancy
    project(flat_rec, flat_points, weights, npart, nprof);
    suppress_zeros_norm(flat_rec, nprof, nbins);
    
    find_difference_profile(diff_prof, flat_rec, flat_profiles, all_bins);
    discr[niter] = discrepancy(diff_prof, nprof, nbins);

    // Cleaning
    for(i = 0; i < nprof; i++) {
        delete[] rparts[i];
    }
    for(i = 0; i < npart; i++) {
        delete[] flat_points[i];
    }
    delete[] rparts, flat_points, flat_rec, discr, diff_prof;
}
