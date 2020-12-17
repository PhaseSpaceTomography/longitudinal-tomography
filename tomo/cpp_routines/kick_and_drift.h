//
// Created by anton on 10/13/20.
//

#ifndef TOMOGRAPHY_KICK_AND_DRIFT_CPP_H
#define TOMOGRAPHY_KICK_AND_DRIFT_CPP_H


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
                        const double acc_kick);

extern "C" void kick_down(const double * __restrict__ dphi,
                          double * __restrict__ denergy,
                          const double rfv1,
                          const double rfv2,
                          const double phi0,
                          const double phi12,
                          const double hratio,
                          const int nr_particles,
                          const double acc_kick);

// "Drift" function.
// Calculates the difference in phase between two macine turns.
// Can be called directly from python.
//  Used in hybrid python/C++ class.
extern "C" void drift_up(double * __restrict__ dphi,
                         const double * __restrict__ denergy,
                         const double drift_coef,
                         const int nr_particles);

extern "C" void drift_down(double * __restrict__ dphi,
                           const double * __restrict__ denergy,
                           const double drift_coef,
                           const int nr_particles);


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
                               const int nparts);

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
        const bool ftn_out);

#endif
