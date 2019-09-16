#include <iostream>
// #include <iomanip>
#include "sin.h"

using namespace vdt;

// Calculating the energy kick at a turn
extern "C" void kick(const double * __restrict__ dphi,
                         double * __restrict__ denergy,
                         const double rfv1,
                         const double rfv2,
                         const double phi0,
                         const double phi12,
                         const double hratio,
                         const int nr_particles,
                         const double acc_kick){
#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        denergy[i] = denergy[i] + rfv1 * fast_sin(dphi[i] + phi0)
                   + rfv2 * fast_sin(hratio * (dphi[i] + phi0 - phi12));

#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        denergy[i] = denergy[i] - acc_kick;
}

// Calculating change in phase for each turn
extern "C" void drift(double * __restrict__ dphi,
                      const double * __restrict__ denergy,
                      const double dphase,
                      const int nr_particles){
#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] -= dphase * denergy[i];
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
                         const double * __restrict__ dphase,// inn
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
    
    // One should put add a check here that the number of turns corresponds
    // to the number of profiles, and then indices in xp and yp arrays.

    // std::cout<<std::fixed<<std::setprecision(10);

    int profile = 0;
    int turn = 0;

    std::cout << "Tracking to profile " << profile + 1 << std::endl;

    while(turn < nturns){
        drift(dphi, denergy, dphase[turn], nparts);

        turn++;

        kick(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn], phi12,
             hratio, nparts, deltaE0[turn]);

        if (turn % dturns == 0){
            profile++;
            
            for(int i=0; i < nparts; i++){
                xp[profile][i] = (dphi[i] + phi0[turn])
                                  / (hnum * omega_rev0[turn] * dtbin)
                                  - xorigin;
            } //for xp

            for(int i=0; i < nparts; i++){
                yp[profile][i] = denergy[i] / dEbin + yat0;
            } //for yp

            std::cout << "Tracking to profile " << profile + 1 << std::endl;
        } //if
    } //while
}
