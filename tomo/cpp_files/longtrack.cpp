 #include <iostream>
 #include <fstream>
 #include <stdlib.h>
 #include <cmath>
 #include <vector>
 #include <string>

 using namespace std;

extern "C"{

    double* calculate_dphi(const double xp[],
                      int xp_len, 
                      double xorigin, int h_num,
                      double omega_rev0, double dtbin,
                      double phi0){
        double * dphi = new double[xp_len];

        for(int i=0; i < xp_len; i++){
            dphi[i] = (xp[i] + xorigin) * h_num * omega_rev0 * dtbin - phi0;
        }

        return dphi;
    }

    double* calculate_denergy(const double yp[], int yp_len,
                          double yat0, double dEbin){
        double * denergy = new double[yp_len];
        for(int i=0; i < yp_len; i++){
            denergy[i] = (yp[i] - yat0) * dEbin;
        }
        return denergy;
    }

    // Returns current turn_now
    int longtrack(double xp[],                  //inout
                  double yp[],                  //inout
                  const double omega_rev0[],    //in
                  const double phi0[],          //in
                  const double c1[],            //in
                  const double turn_time[],     //in
                  const double deltaE0[],       //in
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
                
                #pragma omp paralell for
                for(j=0; j < xp_len; j++){
                    dphi[j] -= c1[turn_now] * denergy[j];
                }
                turn_now++;

                rfv1_at_turn = vrf1 + vrf1dot * turn_time[turn_now];
                rfv2_at_turn = vrf2 + vrf2dot * turn_time[turn_now];
                temp_phi = phi0[turn_now] - phi12;
                
                #pragma omp paralell for
                for(j=0; j < xp_len; j++){
                    denergy[j] += q * (rfv1_at_turn
                                       * sin(dphi[j] + phi0[turn_now])
                                       + rfv2_at_turn
                                       * sin(h_ratio 
                                             * (dphi[j] + temp_phi))
                                       - deltaE0[turn_now]);
                }
            }
        }
        else{
            throw new exception;
        }

        #pragma omp paralell for
        for(i=0; i < xp_len; i++){
            xp[i] = (dphi[i] + phi0[turn_now])
                    / (h_num * omega_rev0[turn_now] * dtbin)
                    - xorigin;
        }
        
        #pragma omp paralell for
        for(i=0; i < xp_len; i++){
            yp[i] = denergy[i] / dEbin + yat0;
        }

        delete[] dphi;
        delete[] denergy;

        return turn_now;
    }
}