#include <iostream>

#include <fstream>
#include "timing.h"
#include "fileIO.h"

#include <string>
#include "sin.h"
#include <cmath>

using namespace std;

extern "C" void gpu_kick(const double * __restrict__ dphi,
                         double * __restrict__ denergy,
                         const double rfv1,
                         const double rfv2,
                         const double phi0,
                         const double phi12,
                         const double hratio,
                         const int nr_particles,
                         const double acc_kick){

    #pragma acc parallel loop device_type(nvidia) vector_length(32)
    for (int i=0; i < nr_particles; i++)    
        denergy[i] = denergy[i] + rfv1 * sin(dphi[i] + phi0)
                     + rfv2 * sin(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
}

extern "C" void cpu_kick(const double * __restrict__ dphi,
                         double * __restrict__ denergy,
                         const double rfv1,
                         const double rfv2,
                         const double phi0,
                         const double phi12,
                         const double hratio,
                         const int nr_particles,
                         const double acc_kick){

    #pragma omp parallel for
    for (int i=0; i < nr_particles; i++)    
        denergy[i] = denergy[i] + rfv1 * vdt::fast_sin(dphi[i] + phi0)
                     + rfv2 * vdt::fast_sin(hratio * (dphi[i] + phi0 - phi12)) - acc_kick;
}

extern "C" void drift(double * __restrict__ dphi,
                      const double * __restrict__ denergy,
                      const double dphase,
                      const int nr_particles){
    #pragma acc parallel loop device_type(nvidia) vector_length(32)
    #pragma omp parallel for
    for (int i = 0; i < nr_particles; i++)
        dphi[i] -= dphase * denergy[i];
}

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
                               const int nparts){
    #pragma acc parallel loop device_type(nvidia) vector_length(32)
    #pragma acc declare present(dphi[:nparts], denergy[:nparts])
    #pragma omp parallel for
    for(int i=0; i < nparts; i++){
        xp[profile][i] = (dphi[i] + phi0) / (hnum * omega_rev0 * dtbin) - xorigin;
        yp[profile][i] = denergy[i] / dEbin + yat0;
    }//for
}

// GPU VERSION
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
                         const double * __restrict__ dphase,    // inn
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
    
    // One should add a check here that the number of turns corresponds
    // to the number of profiles, and then indices in xp and yp arrays.

    // cout<< fixed<< setprecision(10);

    int profile = 0;
    int turn = 0;
    int nprofs = (nturns / dturns) + 1;
    int turn_arr_len = nturns + 1; 

    cout << "Tracking..." << endl;
 #pragma acc data copyin(dphase[:turn_arr_len], rf1v[:turn_arr_len],\
                         rf2v[:turn_arr_len],phi0[:turn_arr_len],\
                         omega_rev0[:turn_arr_len], deltaE0[:turn_arr_len])\
                  copyin(denergy[:nparts], dphi[:nparts])\
                  copyout(xp[:nprofs][:nparts], yp[:nprofs][:nparts])
    {
    while(turn < nturns){
        drift(dphi, denergy, dphase[turn], nparts);
        
        turn++;
        
        gpu_kick(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn], phi12,
                 hratio, nparts, deltaE0[turn]);

        if(turn % dturns == 0){
            
            profile++;
            
            calc_xp_and_yp(xp, yp, denergy, dphi, phi0[turn], hnum,
                           omega_rev0[turn], dtbin, xorigin, dEbin,
                           yat0, profile, nparts);
        }// if
    } //while
    } // data
} //func

// CPU VERSION
extern "C" void kick_and_drift_cpu(
                         double ** __restrict__ xp,             // inn/out
                         double ** __restrict__ yp,             // inn/out
                         double * __restrict__ denergy,         // inn
                         double * __restrict__ dphi,            // inn
                         const double * __restrict__ rf1v,      // inn
                         const double * __restrict__ rf2v,      // inn
                         const double * __restrict__ phi0,      // inn
                         const double * __restrict__ deltaE0,   // inn
                         const double * __restrict__ omega_rev0,// inn
                         const double * __restrict__ dphase,    // inn
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
    
    // One should add a check here that the number of turns corresponds
    // to the number of profiles, and then indices in xp and yp arrays.

    int profile = 0;
    int turn = 0;
    int profile_count = nturns / dturns;

    cout << "Tracking to profile " << profile + 1 << endl;

    while(turn < nturns){
        drift(dphi, denergy, dphase[turn], nparts);
        
        turn++;
        
        cpu_kick(dphi, denergy, rf1v[turn], rf2v[turn], phi0[turn], phi12,
                 hratio, nparts, deltaE0[turn]);   
        
        if (turn % dturns == 0){
            
            profile++;

            calc_xp_and_yp(xp, yp, denergy, dphi, phi0[turn], hnum,
                           omega_rev0[turn], dtbin, xorigin, dEbin,
                           yat0, profile, nparts);

            cout << "Tracking to profile " << profile + 1 << endl;

        } //if
    } //while
}//func


int main(int argc, char const *argv[]){
    const string rdir = "/afs/cern.ch/work/c/cgrindhe/tomography/tomo_v3/unit_tests/resources/C500MidPhaseNoise";
    const string rdir2 = "/afs/cern.ch/work/c/cgrindhe/tomography/lab/cpp_out";
    
    const int NPART = 406272;
    const int NTURNS = 1188;
    const int DTURNS = 12;
    const int NPROFS = 100;

    double** xp = new double*[NPROFS];
    double** yp = new double*[NPROFS];
    for(int i = 0; i < NPROFS; i++){
        xp[i] = new double[NPART];
        yp[i] = new double[NPART];
    }

    for (int i = 0; i < NPROFS; i++){
        for (int j = 0; j < NPART; j++){
            xp[i][j] = 0.0;
            yp[i][j] = 0.0;
        }
    }

    double denergy [NPART];
    double dphi [NPART];
    readDoubleArrayTxt(denergy, NPART, rdir2 + "/denergy.dat");    
    readDoubleArrayTxt(dphi, NPART, rdir2 + "/dphi.dat");

    double rf2v [NTURNS+1];
    double rf1v [NTURNS+1];
    double phi0 [NTURNS+1];
    double deltaE0 [NTURNS+1];
    double omega_rev0 [NTURNS+1];
    double dphase [NTURNS+1];

    for(int i = 0; i < NTURNS+1; i++){
        rf2v[i] = 0.0; 
    }

    readDoubleArrayTxt(rf1v, NTURNS+1, rdir2 + "/rfv1.dat");
    readDoubleArrayTxt(phi0, NTURNS+1, rdir + "/phi0.dat");  
    readDoubleArrayTxt(omega_rev0, NTURNS+1, rdir + "/omegarev0.dat");
    readDoubleArrayTxt(deltaE0, NTURNS+1, rdir + "/deltaE0.dat");
    readDoubleArrayTxt(dphase, NTURNS+1, rdir + "/dphase.dat");

    const double phi12 = 0.3116495273194016;
    const double hratio = 2.0;
    const double hnum = 1.0;
    const double dtbin = 1.9999999999999997e-09;
    const double xorigin = -69.73326295579088;
    const double dEbin = 23340.63328895732;
    const double yat0 = 102.5;

    Timer tm = Timer();

    cout << "All data loaded!" << endl;
    cout << "Crossing fingers...." << endl << endl;

    tm.startClock();
    
    kick_and_drift(xp, yp, denergy, dphi, rf1v, rf2v, phi0,
                   deltaE0, omega_rev0, dphase, phi12, hratio,
                   hnum, dtbin, xorigin, dEbin, yat0, DTURNS,
                   NTURNS, NPART);
    
    auto tdiff = tm.getTime();

    cout << "Success!" << endl;
    cout << "Time spent: " << tdiff  << "ms" << endl;

    cout << endl << endl << "Saving xp..." << endl;
    string out_dir = "/afs/cern.ch/work/c/cgrindhe/tomography/lab/cpp_out";
    save2dArrayTxt(xp, NPART, NPROFS, out_dir + "/xp_cpp.dat");
    cout << "XP saved, saving yp..." << endl;
    save2dArrayTxt(yp, NPART, NPROFS, out_dir + "/yp_cpp.dat");
    cout << "Saving complete!" << endl;

    for(int i = 0; i < NPROFS; i++){
        delete[] xp[i], yp[i];
    }
    delete[] xp, yp;

    return 0;
}