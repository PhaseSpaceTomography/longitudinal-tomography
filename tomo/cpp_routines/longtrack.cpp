#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>
#include "omp.h"

 using namespace std;

extern "C"{

    // Calculating array containing differences between bin in i direction
    double* calculate_dphi(const double * __restrict__  xp,
                      int xp_len, 
                      double xorigin, int h_num,
                      double omega_rev0, double dtbin,
                      double phi0){
        double * dphi = new double[xp_len];
        #pragma omp paralell for num_threads(4)
        for(int i=0; i < xp_len; i++){
            dphi[i] = (xp[i] + xorigin) * h_num * omega_rev0 * dtbin - phi0;
        }

        return dphi;
    }

    // Calculating array containing differences in energy between each bin in j direction.
    double* calculate_denergy(const double * __restrict__ yp, int yp_len,
                          double yat0, double dEbin){
        double * denergy = new double[yp_len];
        #pragma omp paralell for num_threads(4)
        for(int i=0; i < yp_len; i++){
            denergy[i] = (yp[i] - yat0) * dEbin;
        }
        return denergy;
    }

    double calc_denergy_at_turn(double phi0, double dphi, double deltaE0,
                                double rfv1, double rfv2, double hratio,
                                double delta_phi, double q){
        return q * (rfv1 * sin(dphi + phi0)
                    + rfv2 * sin(hratio * (dphi + delta_phi)))
                - deltaE0;
    }

    // Tracking the particles between two profile measurements.
    // The number of machine turns between the measurements is given by nreps
    // The direction is for tracking both forward and backward in time
    // The function will return x and y coordinates of the particles in the bunch (xp, yp)
    // and the last tracked machine turn.
    int longtrack(double * __restrict__ xp,                  //inout
                  double * __restrict__ yp,                  //inout
                  const double  * __restrict__ omega_rev0,    //in
                  const double  * __restrict__ phi0,          //in
                  const double  * __restrict__ c1,            //in
                  const double  * __restrict__ turn_time,     //in
                  const double  * __restrict__ deltaE0,       //in
                  int xp_len, double xorigin, 
                  double dtbin, double dEbin,
                  double yat0, double phi12,
                  int direction, int nreps, 
                  int turn_now, double q,
                  double vrf1, double vrf1dot,
                  double vrf2, double vrf2dot,
                  int h_num, double h_ratio){
        
        double rfv1_at_turn;
        double rfv2_at_turn;
        double temp_phi; 

        double * dphi = calculate_dphi(xp, xp_len, xorigin, h_num,
                                       omega_rev0[turn_now], dtbin,
                                       phi0[turn_now]);
        double * denergy =  calculate_denergy(yp, xp_len, yat0, dEbin);

        int i=0, j=0;
        if(direction > 0){
            for(i=0; i < nreps; i++){
                
                #pragma omp parallel for
                for(j=0; j < xp_len; j++){
                    dphi[j] -= c1[turn_now] * denergy[j];
                }
                turn_now++;

                rfv1_at_turn = vrf1 + vrf1dot * turn_time[turn_now];
                rfv2_at_turn = vrf2 + vrf2dot * turn_time[turn_now];
                temp_phi = phi0[turn_now] - phi12;
                
                #pragma omp parallel for
                for(j=0; j < xp_len; j++){
                    denergy[j] += calc_denergy_at_turn(
                                    phi0[turn_now], dphi[j], deltaE0[turn_now],
                                    rfv1_at_turn, rfv2_at_turn,
                                    h_ratio, temp_phi, q); 
                }
            }
        }
        else{
            for(i=0; i < nreps; i++){
                
                rfv1_at_turn = vrf1 + vrf1dot * turn_time[turn_now];
                rfv2_at_turn = vrf2 + vrf2dot * turn_time[turn_now];
                temp_phi = phi0[turn_now] - phi12;
            
                #pragma omp parallel for
                for(j=0; j < xp_len; j++){
                    denergy[j] -= calc_denergy_at_turn(
                                    phi0[turn_now], dphi[j], deltaE0[turn_now],
                                    rfv1_at_turn, rfv2_at_turn,
                                    h_ratio, temp_phi, q); 
                }
                turn_now--;
                #pragma omp parallel for
                for(j=0; j < xp_len; j++){
                    dphi[j] += c1[turn_now] * denergy[j];
                }
            }
        }

        #pragma omp parallel for
        for(i=0; i < xp_len; i++){
            xp[i] = (dphi[i] + phi0[turn_now])
                    / (h_num * omega_rev0[turn_now] * dtbin)
                    - xorigin;
        }
                
        for(i=0; i < xp_len; i++){
            yp[i] = denergy[i] / dEbin + yat0;
        }
        

        delete[] dphi;
        delete[] denergy;        

        return turn_now;
    }
}