#include "sin.h"

using namespace vdt;

// Calculating the energy kick at a turn
extern "C" void new_kick(const double * __restrict__ dphi,
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