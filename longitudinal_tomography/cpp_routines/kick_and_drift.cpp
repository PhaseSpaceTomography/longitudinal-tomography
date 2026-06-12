/**
 * @file kick_and_drift.cpp
 *
 * @author Anton Lu
 * Contact: anton.lu@cern.ch
 *
 * Functions in pure C/C++ that handles particle tracking (kicking and
 * drifting). Meant to be called by a Python/C++ wrapper.
 */

#define _USE_MATH_DEFINES

#include <iostream>
#include <string>
#include "sin.h"
#include <cmath>
#include "kick_and_drift.h"
#include <atomic>

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
             const real_t acc_kick) {

#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        if (std::is_same<real_t, double>::value)
            denergy[i] += rfv1 * vdt::fast_sin(dphi[i] + phi0)
                        + rfv2 * vdt::fast_sin(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
        else if(std::is_same<real_t, float>::value)
            denergy[i] += rfv1 * vdt::fast_sinf(dphi[i] + phi0)
                        + rfv2 * vdt::fast_sinf(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
}

template <typename real_t>
void kick_down(const real_t *dphi,
               real_t *denergy,
               const real_t rfv1,
               const real_t rfv2,
               const real_t phi0,
               const real_t phi12,
               const real_t hratio,
               const int nr_particles,
               const real_t acc_kick) {

#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        if (std::is_same<real_t, double>::value)
            denergy[i] -= rfv1 * vdt::fast_sin(dphi[i] + phi0)
                        + rfv2 * vdt::fast_sin(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
        else if(std::is_same<real_t, float>::value)
            denergy[i] -= rfv1 * vdt::fast_sinf(dphi[i] + phi0)
                        + rfv2 * vdt::fast_sinf(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
}

template <typename int_t>
void kick_up_int(const int_t *dphi,
             int_t *denergy,
             const int_t rfv1,
             const int_t rfv2,
             const int_t phi0,
             const int_t phi12,
             const int_t hratio,
             const int_t nr_particles,
             const int_t acc_kick,
             const int_t S,
             const int_t G,
             const int_t abs_of_lowest_possible_angle_int,
             const int_t reciprocal_lut_index_factor,
             const int_t *lut
            ) {

    atomic<bool> exception_caught(false);
    exception_ptr first_exception = nullptr;

#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++){
        // If an exception was already caught, skip work
        if (exception_caught.load(memory_order_relaxed)){
            continue;
        }
        try{
            //if (i == 0){
            //    cout << i << ": " << dphi[i] + phi0 << " "<< hratio * (dphi[i] + phi0 - phi12) << " - "<< dphi[i] << " " << phi0 << " " << phi12 << " " << acc_kick << endl;
            //    cout << sin_fixed_point(dphi[i] + phi0, x0_int, dx_int, lut, G) << " " << sin_fixed_point(hratio * (dphi[i] + phi0 - phi12), x0_int, dx_int, lut, G) << endl;
            //}
            denergy[i] += rfv1 * sin_fixed_point(dphi[i] + phi0, abs_of_lowest_possible_angle_int, reciprocal_lut_index_factor, lut, G)/S
                        + rfv2 * sin_fixed_point(hratio * (dphi[i] + phi0 - phi12), abs_of_lowest_possible_angle_int, reciprocal_lut_index_factor, lut, G)/S - acc_kick;
        }
        catch (const exception &e) {
            // Only first thread stores the exception
            bool expected = false;
            if (exception_caught.compare_exchange_strong(expected, true)){
                first_exception = std::current_exception();
            }
        }
    }
    if (exception_caught){
        rethrow_exception(first_exception);
    }
}

template <typename int_t>
void kick_down_int(const int_t *dphi,
               int_t *denergy,
               const int_t rfv1,
               const int_t rfv2,
               const int_t phi0,
               const int_t phi12,
               const int_t hratio,
               const int_t nr_particles,
               const int_t acc_kick,
               const int_t S,
               const int_t G,
               const int_t abs_of_lowest_possible_angle_int,
               const int_t reciprocal_lut_index_factor,
               const int_t *lut
            ) {

    atomic<bool> exception_caught(false);
    exception_ptr first_exception = nullptr;

#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++){
        // If an exception was already caught, skip work
        if (exception_caught.load(memory_order_relaxed)){
            continue;
        }
        try{
            denergy[i] -= ((rfv1 * sin_fixed_point(dphi[i] + phi0, abs_of_lowest_possible_angle_int, reciprocal_lut_index_factor, lut, G)) >> S)
                        + ((rfv2 * sin_fixed_point(hratio * (dphi[i] + phi0 - phi12), abs_of_lowest_possible_angle_int, reciprocal_lut_index_factor, lut, G)) >> S) - acc_kick;
        }
        catch (const exception &e) {
            // Only first thread stores the exception
            bool expected = false;
            if (exception_caught.compare_exchange_strong(expected, true)){
                first_exception = std::current_exception();
            }
        }
    }
    if (exception_caught){
        rethrow_exception(first_exception);
    }
}

// "Drift" function.
// Calculates the difference in phase between two macine turns.
// Can be called directly from python.
//  Used in hybrid python/C++ class.
template <typename real_t>
void drift_up(real_t *dphi,
              const real_t *denergy,
              const real_t drift_coef,
              const int nr_particles) {
#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++){
        dphi[i] -= drift_coef * denergy[i];
    }
}

template <typename real_t>
void drift_down(real_t *dphi,
                const real_t *denergy,
                const real_t drift_coef,
                const int nr_particles) {

#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] += drift_coef * denergy[i];
}

template <typename int_t>
void drift_up_int(int_t *dphi,
              const int_t *denergy,
              const int_t drift_coef,
              const int_t nr_particles,
              const int_t S) {
#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] -= drift_coef * denergy[i] >> S;
}

template <typename int_t>
void drift_down_int(int_t *dphi,
                const int_t *denergy,
                const int_t drift_coef,
                const int_t nr_particles,
                const int_t S) {

#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] += drift_coef * denergy[i] >> S;
}

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
                               const int nparts) {
#pragma omp parallel for
    for (int i = 0; i < nparts; i++) {
        xp[profile][i] = (dphi[i] + phi0) / (hnum * omega_rev0 * dtbin) - xorigin;
        yp[profile][i] = denergy[i] / dEbin + yat0;
    }//for
}

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
                    const std::function<void(int, int)> callback) {
    int profile = rec_prof;
    int turn = rec_prof * dturns + deltaturn;
    if (deltaturn < 0) profile--;

#pragma omp parallel for
    for (int i = 0; i < nparts; i++) {
        xp[profile][i] = dphi[i];
        yp[profile][i] = denergy[i];
    }

    int progress = 0;
    const int total = nturns;
    // Upwards 
    while (turn < nturns) {
        drift_up<real_t>(dphi, denergy, drift_coef[turn], nparts);

        turn++;

        kick_up<real_t>(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn], phi12[turn],
                hratio, nparts, deltaE0[turn]);

        if (turn % dturns == 0) {
            profile++;
#pragma omp parallel for
            for (int i = 0; i < nparts; i++) {
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

    if (profile > 0) {

        // Going back to initial coordinates
#pragma omp parallel for
        for (int i = 0; i < nparts; i++) {
            dphi[i] = xp[rec_prof][i];
            denergy[i] = yp[rec_prof][i];
        }

        // Downwards
        while (turn > 0) {
            kick_down<real_t>(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn],
                      phi12[turn], hratio, nparts, deltaE0[turn]);
            turn--;

            drift_down<real_t>(dphi, denergy, drift_coef[turn], nparts);

            if (turn % dturns == 0) {
                profile--;

#pragma omp parallel for
                for (int i = 0; i < nparts; i++) {
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

template <typename int_t, typename real_t>
void kick_and_drift_int(int_t **xp,             // inn/out
                    int_t **yp,             // inn/out
                    int_t *denergy,         // inn
                    int_t *dphi,            // inn
                    const int_t *rf1v,      // inn
                    const int_t *rf2v,      // inn
                    const int_t *phi0,      // inn
                    const int_t *deltaE0,   // inn
                    const int_t *drift_coef,// inn
                    const int_t *phi12,
                    const int_t hratio,
                    const int_t dturns,
                    const int_t rec_prof,
                    const int_t deltaturn,
                    const int_t nturns,
                    const int_t nparts,
                    const bool ftn_out,
                    const int_t S,
                    const int_t G,
                    const real_t abs_of_lowest_possible_angle,
                    const std::function<void(int, int)> callback) {
    int profile = rec_prof;
    int turn = rec_prof * dturns + deltaturn;

    int_t abs_of_lowest_possible_angle_int = std::ldexp(abs_of_lowest_possible_angle, S);

    real_t reciprocal_lut_index_factor = (1 << (G + 24)) / (2 * M_PI * (1 << S));
    int_t lut[1<<G];
    int_t dx_int = generate_sin_lut(lut, 2 * M_PI, G, S);



    if (deltaturn < 0) profile--;

#pragma omp parallel for
    for (int i = 0; i < nparts; i++) {
        xp[profile][i] = dphi[i];
        yp[profile][i] = denergy[i];
    }

    int progress = 0;
    const int total = nturns;
    // Upwards 
    while (turn < nturns) {
        // cout << turn << " " << nturns << " " << dphi[0] << " " << denergy[0] << " " << drift_coef[turn] << endl;
        drift_up_int<int_t>(dphi, denergy, drift_coef[turn], nparts, S);
        
        turn++;
        // cout << turn << " " << rf1v[turn-1] << " " << rf2v[turn-1] << " " << phi0[turn-1] << " " << phi12[turn-1] << " " << deltaE0[turn-1] << endl;
        kick_up_int<int_t>(dphi, denergy, rf1v[turn-1], rf2v[turn-1], phi0[turn-1], phi12[turn-1],
                hratio, nparts, deltaE0[turn-1], S, G, abs_of_lowest_possible_angle_int, (int_t) reciprocal_lut_index_factor, lut);

        if (turn % dturns == 0) {
            profile++;
#pragma omp parallel for
            for (int i = 0; i < nparts; i++) {
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

    if (profile > 0) {

        // Going back to initial coordinates
#pragma omp parallel for
        for (int i = 0; i < nparts; i++) {
            dphi[i] = xp[rec_prof][i];
            denergy[i] = yp[rec_prof][i];
        }

        // Downwards
        while (turn > 0) {          
            //cout << turn << " - " << rf1v[turn-1] << " " << rf2v[turn-1] << " " << phi0[turn-1] << " " << phi12[turn-1] << " " << deltaE0[turn-1] << endl;
            kick_down_int<int_t>(dphi, denergy, rf1v[turn-1], rf2v[turn-1], phi0[turn-1],
                      phi12[turn-1], hratio, nparts, deltaE0[turn-1], S, G, abs_of_lowest_possible_angle_int, (int_t) reciprocal_lut_index_factor, lut);
            turn--;

            drift_down_int<int_t>(dphi, denergy, drift_coef[turn], nparts, S);

            if (turn % dturns == 0) {
                profile--;

#pragma omp parallel for
                for (int i = 0; i < nparts; i++) {
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

template <typename int_t, typename real_t>
int_t generate_sin_lut(int_t *lut,
                       real_t two_pi_angle,
                       int_t G,
                       int_t S){
    // The increment for the values in the LUT is 2*pi/(2**G)
    real_t dx = std::ldexp(two_pi_angle, -G);
    for (int i = 0; i < (1 << G); i ++){
        real_t x = i*dx;
        lut[i] = (int_t) std::ldexp(sin(x), S);
    }

    int_t dx_int = (int_t) std::ldexp(dx, S);

    if (dx_int <= 0){
        throw range_error("Error in generating the look-up table, `dx_int` <= 0.");
    }

    return dx_int;
}

template <typename int_t>
int_t sin_fixed_point(int_t x_int,
                    int_t abs_of_lowest_possible_angle,
                      int_t reciprocal_lut_index_factor,
                      const int_t *lut,
                      int_t G){
    // instead of dividing by the stepsize between lut entires
    // -> multiply with the reciprocal (scaled by 2**24, which needs to be undone)
    // We add the abs of the lowest possible angle beforehand -> Therefore the value is always possitive
    // and we know 2**G is equal to 2 Pi so we can truncate all the bits above the Gth bit
    // int64_t needed here, because of the scaling regardless if we use 32 or 64 bit otherwise
    int64_t idx = ((x_int + abs_of_lowest_possible_angle) * reciprocal_lut_index_factor) >> 24;
    idx = idx & ((1 << G) - 1);
    // The mask guarantees, we don't need to check for over or underflow -> No need for checks
    return lut[idx];
}

template int32_t sin_fixed_point(int32_t x_int,
                                 int32_t abs_of_lowest_possible_angle,
                                 int32_t reciprocal_lut_index_factor,
                                 const int32_t *lut,
                                 int32_t G);
template int64_t sin_fixed_point(int64_t x_int,
                                 int64_t abs_of_lowest_possible_angle,
                                 int64_t reciprocal_lut_index_factor,
                                 const int64_t *lut,
                                 int64_t G);

template int32_t generate_sin_lut(int32_t *lut,
                                  float two_pi_angle,
                                  int32_t G,
                                  int32_t S);
template int64_t generate_sin_lut(int64_t *lut,
                                  float two_pi_angle,
                                  int64_t G,
                                  int64_t S);
template int32_t generate_sin_lut(int32_t *lut,
                                  double two_pi_angle,
                                  int32_t G,
                                  int32_t S);
template int64_t generate_sin_lut(int64_t *lut,
                                  double two_pi_angle,
                                  int64_t G,
                                  int64_t S);

template void kick_and_drift_int(int32_t **xp,
                             int32_t **yp,
                             int32_t *denergy,
                             int32_t *dphi,
                             const int32_t *rf1v,
                             const int32_t *rf2v,
                             const int32_t *phi0,
                             const int32_t *deltaE0,
                             const int32_t *drift_coef,
                             const int32_t *phi12,
                             const int32_t hratio,
                             const int32_t dturns,
                             const int32_t rec_prof,
                             const int32_t deltaturn,
                             const int32_t nturns,
                             const int32_t nparts,
                             const bool ftn_out,
                             const int32_t S,
                             const int32_t G,
                             const float abs_of_lowest_possible_angle,
                             const std::function<void(int, int)> callback);

template void kick_and_drift_int(int32_t **xp,
                             int32_t **yp,
                             int32_t *denergy,
                             int32_t *dphi,
                             const int32_t *rf1v,
                             const int32_t *rf2v,
                             const int32_t *phi0,
                             const int32_t *deltaE0,
                             const int32_t *drift_coef,
                             const int32_t *phi12,
                             const int32_t hratio,
                             const int32_t dturns,
                             const int32_t rec_prof,
                             const int32_t deltaturn,
                             const int32_t nturns,
                             const int32_t nparts,
                             const bool ftn_out,
                             const int32_t S,
                             const int32_t G,
                             const double abs_of_lowest_possible_angle,
                             const std::function<void(int, int)> callback);

template void kick_and_drift_int(int64_t **xp,
                             int64_t **yp,
                             int64_t *denergy,
                             int64_t *dphi,
                             const int64_t *rf1v,
                             const int64_t *rf2v,
                             const int64_t *phi0,
                             const int64_t *deltaE0,
                             const int64_t *drift_coef,
                             const int64_t *phi12,
                             const int64_t hratio,
                             const int64_t dturns,
                             const int64_t rec_prof,
                             const int64_t deltaturn,
                             const int64_t nturns,
                             const int64_t nparts,
                             const bool ftn_out,
                             const int64_t S,
                             const int64_t G,
                             const float abs_of_lowest_possible_angle,
                             const std::function<void(int, int)> callback);

template void kick_and_drift_int(int64_t **xp,
                             int64_t **yp,
                             int64_t *denergy,
                             int64_t *dphi,
                             const int64_t *rf1v,
                             const int64_t *rf2v,
                             const int64_t *phi0,
                             const int64_t *deltaE0,
                             const int64_t *drift_coef,
                             const int64_t *phi12,
                             const int64_t hratio,
                             const int64_t dturns,
                             const int64_t rec_prof,
                             const int64_t deltaturn,
                             const int64_t nturns,
                             const int64_t nparts,
                             const bool ftn_out,
                             const int64_t S,
                             const int64_t G,
                             const double abs_of_lowest_possible_angle,
                             const std::function<void(int, int)> callback);

template void kick_and_drift(double **xp,
                             double **yp,
                             double *denergy,
                             double *dphi,
                             const double *rf1v,
                             const double *rf2v,
                             const double *phi0,
                             const double *deltaE0,
                             const double *drift_coef,
                             const double *phi12,
                             const double hratio,
                             const int dturns,
                             const int rec_prof,
                             const int deltaturn,
                             const int nturns,
                             const int nparts,
                             const bool ftn_out,
                             const std::function<void(int, int)> callback);

template void kick_and_drift(float **xp,
                             float **yp,
                             float *denergy,
                             float *dphi,
                             const float *rf1v,
                             const float *rf2v,
                             const float *phi0,
                             const float *deltaE0,
                             const float *drift_coef,
                             const float *phi12,
                             const float hratio,
                             const int dturns,
                             const int rec_prof,
                             const int deltaturn,
                             const int nturns,
                             const int nparts,
                             const bool ftn_out,
                             const std::function<void(int, int)> callback);

template void kick_up(const double *dphi,
                      double *denergy,
                      const double rfv1,
                      const double rfv2,
                      const double phi0,
                      const double phi12,
                      const double hratio,
                      const int nr_particles,
                      const double acc_kick);

template void kick_up(const float *dphi,
                      float *denergy,
                      const float rfv1,
                      const float rfv2,
                      const float phi0,
                      const float phi12,
                      const float hratio,
                      const int nr_particles,
                      const float acc_kick);

template void kick_up_int(const int32_t *dphi,
                      int32_t *denergy,
                      const int32_t rfv1,
                      const int32_t rfv2,
                      const int32_t phi0,
                      const int32_t phi12,
                      const int32_t hratio,
                      const int32_t nr_particles,
                      const int32_t acc_kick,
                      const int32_t S,
                      const int32_t G,
                      const int32_t abs_of_lowest_possible_angle_int,
                      const int32_t reciprocal_lut_index_factor,
                      const int32_t *lut);

template void kick_up_int(const int64_t *dphi,
                      int64_t *denergy,
                      const int64_t rfv1,
                      const int64_t rfv2,
                      const int64_t phi0,
                      const int64_t phi12,
                      const int64_t hratio,
                      const int64_t nr_particles,
                      const int64_t acc_kick,
                      const int64_t S,
                      const int64_t G,
                      const int64_t abs_of_lowest_possible_angle_int,
                      const int64_t reciprocal_lut_index_factor,
                      const int64_t *lut);

template void kick_down(const double *dphi,
                        double *denergy,
                        const double rfv1,
                        const double rfv2,
                        const double phi0,
                        const double phi12,
                        const double hratio,
                        const int nr_particles,
                        const double acc_kick);

template void kick_down(const float *dphi,
                        float *denergy,
                        const float rfv1,
                        const float rfv2,
                        const float phi0,
                        const float phi12,
                        const float hratio,
                        const int nr_particles,
                        const float acc_kick);

template void kick_down_int(const int32_t *dphi,
                      int32_t *denergy,
                      const int32_t rfv1,
                      const int32_t rfv2,
                      const int32_t phi0,
                      const int32_t phi12,
                      const int32_t hratio,
                      const int32_t nr_particles,
                      const int32_t acc_kick,
                      const int32_t S,
                      const int32_t G,
                      const int32_t abs_of_lowest_possible_angle_int,
                      const int32_t reciprocal_lut_index_factor,
                      const int32_t *lut);

template void kick_down_int(const int64_t *dphi,
                      int64_t *denergy,
                      const int64_t rfv1,
                      const int64_t rfv2,
                      const int64_t phi0,
                      const int64_t phi12,
                      const int64_t hratio,
                      const int64_t nr_particles,
                      const int64_t acc_kick,
                      const int64_t S,
                      const int64_t G,
                      const int64_t abs_of_lowest_possible_angle_int,
                      const int64_t reciprocal_lut_index_factor,
                      const int64_t *lut);

template void drift_up(double *dphi,
                       const double *denergy,
                       const double drift_coef,
                       const int nr_particles);

template void drift_up(float *dphi,
                       const float *denergy,
                       const float drift_coef,
                       const int nr_particles);

template void drift_up_int(int32_t *dphi,
                       const int32_t *denergy,
                       const int32_t drift_coef,
                       const int32_t nr_particles,
                       const int32_t S);

template void drift_up_int(int64_t *dphi,
                       const int64_t *denergy,
                       const int64_t drift_coef,
                       const int64_t nr_particles,
                       const int64_t S);

template void drift_down(double *dphi,
                         const double *denergy,
                         const double drift_coef,
                         const int nr_particles);

template void drift_down(float *dphi,
                         const float *denergy,
                         const float drift_coef,
                         const int nr_particles);

template void drift_down_int(int32_t *dphi,
                       const int32_t *denergy,
                       const int32_t drift_coef,
                       const int32_t nr_particles,
                       const int32_t S);

template void drift_down_int(int64_t *dphi,
                       const int64_t *denergy,
                       const int64_t drift_coef,
                       const int64_t nr_particles,
                       const int64_t S);