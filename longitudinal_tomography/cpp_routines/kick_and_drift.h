/**
 * @file kick_and_drift.h
 *
 * @author Anton Lu (anton.lu@cern.ch)
 *
 * Headers for kick_and_drift code for the convenience of not having to
 * put the function definitions in proper order.
 */
#ifndef TOMOGRAPHY_KICK_AND_DRIFT_CPP_H
#define TOMOGRAPHY_KICK_AND_DRIFT_CPP_H

#include <functional>


using namespace std;

// Calculates the difference in energy between two machine turns.
// Uses BLonD fast_sin function.
// Can be called directly from python.
//  Used in hybrid python/C++ class.
extern "C" void kick_up(const double *dphi,
                        double *denergy,
                        const double rfv1,
                        const double rfv2,
                        const double phi0,
                        const double phi12,
                        const double hratio,
                        const int nr_particles,
                        const double acc_kick);

extern "C" void kick_down(const double *dphi,
                          double *denergy,
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
extern "C" void drift_up(double *dphi,
                         const double *denergy,
                         const double drift_coef,
                         const int nr_particles);

extern "C" void drift_down(double *dphi,
                           const double *denergy,
                           const double drift_coef,
                           const int nr_particles);


// Calculates X and Y coordinates for particles based on a given
//  phase and energy.
// Can be called directly from python.
extern "C" void calc_xp_and_yp(double **xp,           // inn/out
                               double **yp,           // inn/out
                               const double *denergy, // inn
                               const double *dphi,    // inn
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
        double **xp,             // inn/out
        double **yp,             // inn/out
        double *denergy,         // inn
        double *dphi,            // inn
        const double *rf1v,      // inn
        const double *rf2v,      // inn
        const double *phi0,      // inn
        const double *deltaE0,   // inn
        const double *drift_coef,// inn
        const double *phi12,
        const double hratio,
        const int dturns,
        const int rec_prof,
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::function<void(int, int)> callback
);

#endif
