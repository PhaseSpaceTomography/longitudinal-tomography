#include <iostream>
#include <string>
#include "sin.h"
#include <cmath>

using namespace std;

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
// Can be called directly from python.
//  Used in hybrid python/C++ class.
extern "C" void drift_up(double * __restrict__ dphi,
                         const double * __restrict__ denergy,
                         const double drift_coef,
                         const int nr_particles){
    #pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] -= drift_coef * denergy[i];
}

extern "C" void drift_down(double * __restrict__ dphi,
                           const double * __restrict__ denergy,
                           const double drift_coef,
                           const int nr_particles){

    #pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] += drift_coef * denergy[i];
}


// Calculates X and Y coordinates for particles based on a given
//  phase and energy.
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
    #pragma omp parallel for
    for(int i=0; i < nparts; i++){
        xp[profile][i] = (dphi[i] + phi0) / (hnum * omega_rev0 * dtbin) - xorigin;
        yp[profile][i] = denergy[i] / dEbin + yat0;
    }//for
}

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
                         const double * __restrict__ drift_coef,// inn
                         const double phi12,
                         const double hratio,
                         const int dturns,
                         const int rec_prof,
                         const int nturns,
                         const int nparts,
                         const bool ftn_out){
    int profile = rec_prof;
    int turn = rec_prof * dturns;

    #pragma omp parallel for
    for(int i=0; i < nparts; i++){
        xp[profile][i] = dphi[i];
        yp[profile][i] = denergy[i];
    }

    // Upwards 
    while(turn < nturns){
        drift_up(dphi, denergy, drift_coef[turn], nparts);
        
        turn++;
        
        kick_up(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn], phi12,
                hratio, nparts, deltaE0[turn]);
        
        if (turn % dturns == 0){
            profile++;
            #pragma omp parallel for
            for(int i=0; i < nparts; i++){
                xp[profile][i] = dphi[i];
                yp[profile][i] = denergy[i];
            }
            if (ftn_out)
                std::cout << " Tracking from time slice  "
                          << rec_prof + 1 << " to  " << profile + 1
                          << ",   0.000% went outside the image width."
                          << std::endl;
        } //if
    } //while

    profile = rec_prof;
    turn = rec_prof * dturns;

    if (profile > 0){

        // Going back to initial coordinates
        #pragma omp parallel for
        for(int i=0; i < nparts; i++){
            dphi[i] = xp[rec_prof][i];
            denergy[i] = yp[rec_prof][i];
        }

        // Downwards
        while(turn > 0){
            kick_down(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn],
                      phi12, hratio, nparts, deltaE0[turn]);
            turn--;
            
            drift_down(dphi, denergy, drift_coef[turn], nparts);
            
            if (turn % dturns == 0){
                profile--;
                
                #pragma omp parallel for
                for(int i=0; i < nparts; i++){
                    xp[profile][i] = dphi[i];
                    yp[profile][i] = denergy[i];
                }
                if (ftn_out)
                    std::cout << " Tracking from time slice  "
                              << rec_prof + 1 << " to  " << profile + 1
                              << ",   0.000% went outside the image width."
                              << std::endl;
            }
        
        }//while
    }

}//end func
