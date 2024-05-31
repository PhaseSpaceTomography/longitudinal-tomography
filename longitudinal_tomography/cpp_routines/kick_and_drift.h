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
template <typename real_t>
void kick_up(const real_t *dphi,
             real_t *denergy,
             const real_t rfv1,
             const real_t rfv2,
             const real_t phi0,
             const real_t phi12,
             const real_t hratio,
             const int nr_particles,
             const real_t acc_kick);

template <typename real_t>
void kick_down(const real_t *dphi,
               real_t *denergy,
               const real_t rfv1,
               const real_t rfv2,
               const real_t phi0,
               const real_t phi12,
               const real_t hratio,
               const int nr_particles,
               const real_t acc_kick);

// "Drift" function.
// Calculates the difference in phase between two macine turns.
// Can be called directly from python.
//  Used in hybrid python/C++ class.
template <typename real_t>
void drift_up(real_t *dphi,
              const real_t *denergy,
              const real_t drift_coef,
              const int nr_particles);

template <typename real_t>
void drift_down(real_t *dphi,
                const real_t *denergy,
                const real_t drift_coef,
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

template <typename real_t>
void kick_and_drift(real_t **xp,             // inn/out
                    real_t **yp,             // inn/out
                    real_t *denergy,         // inn
                    real_t *dphi,            // inn
                    const real_t *rf1v,      // inn
                    const real_t *rf2v,      // inn
                    const real_t *phi0,      // inn
                    const real_t *deltaE0,   // inn
                    const real_t *drift_coef,// inn
                    const real_t *phi12,
                    const real_t hratio,
                    const int dturns,
                    const int rec_prof,
                    const int deltaturn,
                    const int nturns,
                    const int nparts,
                    const bool ftn_out,
                    const std::function<void(int, int)> callback
);

#endif
