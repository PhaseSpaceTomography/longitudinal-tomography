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
template <typename T>
void kick_up(const T *dphi,
                        T *denergy,
                        const T rfv1,
                        const T rfv2,
                        const T phi0,
                        const T phi12,
                        const T hratio,
                        const int nr_particles,
                        const T acc_kick);

template <typename T>
void kick_down(const T *dphi,
                        T *denergy,
                        const T rfv1,
                        const T rfv2,
                        const T phi0,
                        const T phi12,
                        const T hratio,
                        const int nr_particles,
                        const T acc_kick);

// "Drift" function.
// Calculates the difference in phase between two macine turns.
// Can be called directly from python.
//  Used in hybrid python/C++ class.
template <typename T>
void drift_up(T *dphi,
                         const T *denergy,
                         const T drift_coef,
                         const int nr_particles);

template <typename T>
void drift_down(T *dphi,
                           const T *denergy,
                           const T drift_coef,
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

template <typename T>
void kick_and_drift(
        T **xp,             // inn/out
        T **yp,             // inn/out
        T *denergy,         // inn
        T *dphi,            // inn
        const T *rf1v,      // inn
        const T *rf2v,      // inn
        const T *phi0,      // inn
        const T *deltaE0,   // inn
        const T *drift_coef,// inn
        const T *phi12,
        const T hratio,
        const int dturns,
        const int rec_prof,
        const int deltaturn,
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::function<void(int, int)> callback
);

#endif
