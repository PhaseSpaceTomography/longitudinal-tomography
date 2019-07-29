 #include <iostream>
 #include <fstream>
 #include <stdlib.h>
 #include <cmath>
 #include <string>
 #include <omp.h>

 using namespace std;

extern "C"{

    // Function for finding the sum of an array.
    int sum_array_recursive(int arr[], int n){
        if(n <= 0)
            return 0;
        return (sum_array_recursive(arr, n - 1) + arr[n - 1]);
    }

    // Function for calculating one weight factor, and is called by the weight_factor_array function.
    // The function takes a segment of the xp array and saves the coordinates as integers
    // to the mapsi array. The values in the mapsi index will be the bins used in the tomography.
    // the function also saves the weight (number of particles) in each bin to the array mapsw.
    int one_wf(int npt,
               int * __restrict__ mapsi, // inout
               int * __restrict__ mapsw, // inout
               bool * __restrict__ xlog, // inout
               int * __restrict__ xnumb, // inout
               int * __restrict__ xvec,  // inout
               int profile_length,
               int fmlistlength,
               const double * __restrict__ xp_segment, // in
               int submap){

        int isout = 0;
        int icount = 0;
        int start_index = submap * 16;

        for(int i=0; i < npt; i++){
            xlog[i] = true;
            xnumb[i] = 0;
            xvec[i] = ceil(xp_segment[i]);
        }

        for (int i=0; i<npt; i++){
            if(xlog[i]){
                int xet = xvec[i];
                int logsum = 0;
                
                for(int j=0; j < npt; j++){
                    if(xvec[j] == xet & xlog[j]){
                        xnumb[j] = 1;
                    }
                    if(xvec[j] == xet & xlog[j]){
                        xlog[j] = false;
                    }
                }

                if(xet < 1 | xet > profile_length){
                    isout++;
                }
                else{
                    if (icount < fmlistlength){
                        mapsi[start_index + icount] = xet - 1;
                        mapsw[start_index + icount] = sum_array_recursive(xnumb, fmlistlength);

                        for(int j=0; j < npt; j++)
                            xnumb[j] = 0;
                    }
                    else{
                         exit(EXIT_FAILURE);
                    }
                    icount++;
                }
            }
        }
        return isout;
    }

    // Iterates through all bins in profile to fill maps with submap indexes,
    //   mapsi with bin indexes and mapsw with map weights.
    int weight_factor_array(const double * __restrict__ xp, //In
                            const int * __restrict__ jmin,  //In
                            const int * __restrict__ jmax,  //In
                            int * __restrict__ maps,        //Out
                            int * __restrict__ mapsi,       //Out
                            int * __restrict__ mapsw,       //Out
                            int imin,
                            int imax,
                            int npt,
                            int profile_length,
                            int fmlistlength,
                            int submap){                    // out
        int ioffset = 0;
        int isout = 0;
        int uplim = 0;
        int lowlim = 0;
        double* xp_segment = new double[fmlistlength];

        // to be used in calc_one_wf().
        // decleared here for efficiency.
        bool* xlog = new bool[npt];
        int* xnumb = new int[npt];
        int* xvec = new int[npt];
        // -----------------------------

        for(int i=imin; i < imax + 1; i++){
            for(int j=jmin[i]; j < jmax[i]; j++){
                maps[(i * profile_length) + j] = submap;
                lowlim = (j - jmin[i]) * npt + ioffset;
                uplim = (j - jmin[i] + 1) * npt + ioffset;

                // Filling up xp_segment
                int index = 0;
                for(int l=lowlim; l < uplim; l++){
                    double temp = xp[l];
                     xp_segment[index] = xp[l];
                     index++;
                }

                isout += one_wf(npt, mapsi, mapsw, xlog, xnumb,
                                xvec, profile_length, fmlistlength,
                                xp_segment, submap);
                submap++;
            }
            ioffset = uplim;
        }

        delete[] xp_segment;
        delete[] xlog;
        delete[] xnumb;
        delete[] xvec;

        return isout;
    }

    // Creates the first map, which has homogeneously distributed particles.
    // This means that all submaps have npt particles.
    int first_map(const int * __restrict__ jmin, //In
                  const int * __restrict__ jmax, //In
                  int * __restrict__ maps,       //In/Out
                  int * __restrict__ mapsi,      //In/Out
                  int * __restrict__ mapsw,      //In/Out
                  int imin,
                  int imax,
                  int npt,
                  int profile_length){

        int array_depth = npt;
        int submap = 0;
        int index = 0;
        for(int i=imin; i < imax + 1; i++){
            for(int j=jmin[i]; j < jmax[i]; j++){
                index = (i * profile_length) + j;
                maps[index] = submap;
                mapsi[submap * array_depth] = i;
                mapsw[submap * array_depth] = npt;
                submap++;
            }
        }
        return submap;
    }
}