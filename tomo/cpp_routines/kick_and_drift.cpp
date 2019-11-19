#include <iostream>
#include <string>
#include "sin.h"
#include <cmath>

using namespace std;

// "Kick" function for GPU accelerated version.
// Calculates the difference in energy between two machine turns.
// Uses native C++ sinus function.
void kick_gpu(const double * __restrict__ dphi,
              double * __restrict__ denergy,
              const double rfv1,
              const double rfv2,
              const double phi0,
              const double phi12,
              const double hratio,
              const int nr_particles,
              const double acc_kick){

    #pragma acc parallel loop device_type(nvidia) vector_length(32)
    for (int i=0; i < nr_particles; i++)    
        denergy[i] = denergy[i] + rfv1 * sin(dphi[i] + phi0)
                     + rfv2 * sin(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
}

// "Kick" function for CPU accelerated version.
// Calculates the difference in energy between two machine turns.
// Uses BLonD fast_sin function.
// Can be called directly from python.
//  Used in hybrid python/C++ class.
extern "C" void kick_up(const double * __restrict__ dphi,
                        double * __restrict__ denergy,
                        const double rfv1,
                        const double rfv2,
                        const double phi0,
                        const double phi12,
                        const double hratio,
                        const int nr_particles,
                        const double acc_kick){

    #pragma omp parallel for
    for (int i=0; i < nr_particles; i++)    
        denergy[i] += rfv1 * vdt::fast_sin(dphi[i] + phi0)
                     + rfv2 * vdt::fast_sin(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
}

extern "C" void kick_down(const double * __restrict__ dphi,
                          double * __restrict__ denergy,
                          const double rfv1,
                          const double rfv2,
                          const double phi0,
                          const double phi12,
                          const double hratio,
                          const int nr_particles,
                          const double acc_kick){

    #pragma omp parallel for
    for (int i=0; i < nr_particles; i++)    
        denergy[i] -= rfv1 * vdt::fast_sin(dphi[i] + phi0)
                     + rfv2 * vdt::fast_sin(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
}

// "Drift" function.
// Calculates the difference in phase between two macine turns.
// Used for both GPU and CPU version.
// Can be called directly from python.
//  Used in hybrid python/C++ class.
extern "C" void drift_up(double * __restrict__ dphi,
                         const double * __restrict__ denergy,
                         const double dphase,
                         const int nr_particles){

    #pragma acc parallel loop device_type(nvidia) vector_length(32)
    #pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] -= dphase * denergy[i];
}

extern "C" void drift_down(double * __restrict__ dphi,
                           const double * __restrict__ denergy,
                           const double dphase,
                           const int nr_particles){

    #pragma acc parallel loop device_type(nvidia) vector_length(32)
    #pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] += dphase * denergy[i];
}


// Calculates X and Y coordinates for particles based on a given
//  phase and energy.
// Used for both GPU and CPU version.
// Can be called directly from python.
extern "C" void calc_xp_and_yp(double ** __restrict__ xp,           // inn/out
                               double ** __restrict__ yp,           // inn/out
                               const double * __restrict__ denergy, // inn
                               const double * __restrict__ dphi,    // inn
                               const double phi0,
                               const double hnum,
                               const double omega_rev0,
                               const double dtbin,
                               const double xorigin,
                               const double dEbin,
                               const double yat0,
                               const int profile,
                               const int nparts){

    #pragma acc parallel loop device_type(nvidia) vector_length(32)
    #pragma acc declare present(dphi[:nparts], denergy[:nparts])
    #pragma omp parallel for
    for(int i=0; i < nparts; i++){
        xp[profile][i] = (dphi[i] + phi0) / (hnum * omega_rev0 * dtbin) - xorigin;
        yp[profile][i] = denergy[i] / dEbin + yat0;
    }//for
}

// GPU version of Kick and drift function.
// Accelerated with OpenACC.
// The homogeneous particle distribution should be
//  calculated before this function.
// Takes arrays of denergy and dphi with initial values,
//  calculated at forehand.  
extern "C" void kick_and_drift_gpu(
                         double ** __restrict__ xp,             // inn/out
                         double ** __restrict__ yp,             // inn/out
                         double * __restrict__ denergy,         // inn
                         double * __restrict__ dphi,            // inn
                         const double * __restrict__ rf1v,      // inn
                         const double * __restrict__ rf2v,      // inn
                         const double * __restrict__ phi0,      // inn
                         const double * __restrict__ deltaE0,   // inn
                         const double * __restrict__ omega_rev0,// inn
                         const double * __restrict__ dphase,    // inn
                         const double phi12,
                         const double hratio,
                         const double hnum,
                         const double dtbin,
                         const double xorigin,
                         const double dEbin,
                         const double yat0,
                         const int dturns,
                         const int nturns,
                         const int nparts){
    
    // One should add a check here that the number of turns corresponds
    // to the number of profiles, and then indices in xp and yp arrays.

    // cout<< fixed<< setprecision(10);

    int profile = 0;
    int turn = 0;
    int nprofs = (nturns / dturns) + 1;
    int turn_arr_len = nturns + 1; 

    cout << "Tracking..." << endl;
 #pragma acc data copyin(dphase[:turn_arr_len], rf1v[:turn_arr_len],\
                         rf2v[:turn_arr_len],phi0[:turn_arr_len],\
                         omega_rev0[:turn_arr_len], deltaE0[:turn_arr_len])\
                  copyin(denergy[:nparts], dphi[:nparts])\
                  copyout(xp[:nprofs][:nparts], yp[:nprofs][:nparts])
    {
    while(turn < nturns){
        drift_up(dphi, denergy, dphase[turn], nparts);
        
        turn++;
        
        kick_gpu(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn], phi12,
                 hratio, nparts, deltaE0[turn]);

        if(turn % dturns == 0){
            
            profile++;
            
            calc_xp_and_yp(xp, yp, denergy, dphi, phi0[turn], hnum,
                           omega_rev0[turn], dtbin, xorigin, dEbin,
                           yat0, profile, nparts);
        }// if
    } // while
    } // data
} // func

// CPU version of Kick and drift function.
// Accelerated with OpenMP
// The homogeneous particle distribution should be
//  calculated before this function.
// Takes arrays of denergy and dphi with initial values,
//  calculated at forehand.  
extern "C" void kick_and_drift(
                         double ** __restrict__ xp,             // inn/out
                         double ** __restrict__ yp,             // inn/out
                         double * __restrict__ denergy,         // inn
                         double * __restrict__ dphi,            // inn
                         const double * __restrict__ rf1v,      // inn
                         const double * __restrict__ rf2v,      // inn
                         const double * __restrict__ phi0,      // inn
                         const double * __restrict__ deltaE0,   // inn
                         const double * __restrict__ omega_rev0,// inn
                         const double * __restrict__ dphase,    // inn
                         const double phi12,
                         const double hratio,
                         const double hnum,
                         const double dtbin,
                         const double xorigin,
                         const double dEbin,
                         const double yat0,
                         const int dturns,
                         const int nturns,
                         const int nparts){
    
    // One should add a check here that the number of turns corresponds
    // to the number of profiles, and then indices in xp and yp arrays.

    int profile = 0;
    int turn = 0;

    cout << "Tracking to profile " << profile + 1 << endl;

    while(turn < nturns){
        drift_up(dphi, denergy, dphase[turn], nparts);
        
        turn++;
        
        kick_up(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn], phi12,
             hratio, nparts, deltaE0[turn]);   
        
        if (turn % dturns == 0){
            
            profile++;

            calc_xp_and_yp(xp, yp, denergy, dphi, phi0[turn], hnum,
                           omega_rev0[turn], dtbin, xorigin, dEbin,
                           yat0, profile, nparts);

            cout << "Tracking to profile " << profile + 1 << endl;

        } //if
    } //while
}//func
