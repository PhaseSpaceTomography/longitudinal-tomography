
// Calculating change in phase for each turn
extern "C" void new_drift(double * __restrict__ dphi,
                          const double * __restrict__ denergy,
                          const double dphase,
                          const int nr_particles){
#pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] -= dphase * denergy[i];
}
