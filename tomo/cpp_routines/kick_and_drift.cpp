#include <iostream>
#include <string>
#include "sin.h"
#include <cmath>
#include "kick_and_drift.h"

using namespace std;

// Calculates the difference in energy between two machine turns.
// Uses BLonD fast_sin function.
// Can be called directly from python.
//  Used in hybrid python/C++ class.
extern "C" void kick_up(const double * dphi,
                        double * denergy,
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

extern "C" void kick_down(const double * dphi,
                          double * denergy,
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
extern "C" void drift_up(double * dphi,
                         const double * denergy,
                         const double drift_coef,
                         const int nr_particles){
    #pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] -= drift_coef * denergy[i];
}

extern "C" void drift_down(double * dphi,
                           const double * denergy,
                           const double drift_coef,
                           const int nr_particles){

    #pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] += drift_coef * denergy[i];
}


// Calculates X and Y coordinates for particles based on a given
//  phase and energy.
// Can be called directly from python.
extern "C" void calc_xp_and_yp(double ** xp,           // inn/out
                               double ** yp,           // inn/out
                               const double * denergy, // inn
                               const double * dphi,    // inn
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
                         double ** xp,             // inn/out
                         double ** yp,             // inn/out
                         double * denergy,         // inn
                         double * dphi,            // inn
                         const double * rf1v,      // inn
                         const double * rf2v,      // inn
                         const double * phi0,      // inn
                         const double * deltaE0,   // inn
                         const double * drift_coef,// inn
                         const double * phi12,
                         const double hratio,
                         const int dturns,
                         const int rec_prof,
                         const int nturns,
                         const int nparts,
                         const bool ftn_out,
                         const std::function<void(int, int)> callback){
    int profile = rec_prof;
    int turn = rec_prof * dturns;

    #pragma omp parallel for
    for(int i=0; i < nparts; i++){
        xp[profile][i] = dphi[i];
        yp[profile][i] = denergy[i];
    }

    int progress = 0;
    const int total = nturns;
    // Upwards 
    while(turn < nturns){

        #pragma omp parallel for
        for (int i = 0; i < nparts; i++)
            dphi[i] -= drift_coef[turn] * denergy[i];

        turn++;

        #pragma omp parallel for
        for (int i=0; i < nparts; i++)
            denergy[i] += rf1v[turn] * vdt::fast_sin(dphi[i] + phi0[turn])
                          + rf2v[turn] * vdt::fast_sin(hratio * (dphi[i] + phi0[turn] - phi12[turn])) - deltaE0[turn];

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
        callback(++progress, total);
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

            #pragma omp parallel for
            for (int i=0; i < nparts; i++)
                denergy[i] -= rf1v[turn] * vdt::fast_sin(dphi[i] + phi0[turn])
                              + rf2v[turn] * vdt::fast_sin(hratio * (dphi[i] + phi0[turn] - phi12[turn])) - deltaE0[turn];
            turn--;

            #pragma omp parallel for
            for (int i = 0; i < nparts; i++)
                dphi[i] += drift_coef[turn] * denergy[i];

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
        callback(++progress, total);
        }//while
    }

}//end func
