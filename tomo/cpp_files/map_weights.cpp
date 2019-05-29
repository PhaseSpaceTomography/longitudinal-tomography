 #include <iostream>
 #include <fstream>
 #include <stdlib.h>
 #include <cmath>
 #include <string>

 using namespace std;

extern "C"{

    int sum_array_recursive(int arr[], int n){
        if(n <= 0)
            return 0;
        return (sum_array_recursive(arr, n - 1) + arr[n - 1]);
    }

    int one_wf(int npt,
               int mapsi[],         // inout
               int mapsw[],         // inout
               bool xlog[],         // inout
               int xnumb[],         // inout
               int xvec[],          // inout
               int profile_length,
               int fmlistlength,
               double xp_segment[], // in
               int map_nr){

        int isout = 0;
        int icount = 0;
        int start_index = map_nr * 16;

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
                        int temp = sum_array_recursive(xnumb, fmlistlength);
                        mapsw[start_index + icount] = temp;
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

    int weight_factor_array(double xp[],    //In
                        int jmin[],     //In
                        int jmax[],     //In
                        int maps[],     //Out
                        int mapsi[],    //Out
                        int mapsw[],    //Out
                        int jmin_len,
                        int imin,
                        int imax,
                        int npt,
                        int profile_length,
                        int fmlistlength,
                        int number_of_maps,
                        int submap){   // out
        int ioffset = 0;
        int isout = 0;
        int uplim = 0;
        int lowlim = 0;
        double* xp_segment = new double[fmlistlength];
        //maps = new int[profile_length * profile_length];
        //mapsi = new int[number_of_maps * fmlistlength];
        //mapsw = new int[number_of_maps * fmlistlength];

        // Initializing arrays
        for(int i=0; i < fmlistlength; i++){
            xp_segment[i] = 0;
        }
        for(int i=0; i < number_of_maps * fmlistlength; i++){
            mapsi[i] = -1;
            mapsw[i] = 0;
        }
        for(int i=0; i < profile_length * profile_length; i++){
            maps[i] = 0;
        }

        // to be used in calc_one_wf().
        // decleared here for efficiency.
        bool* xlog = new bool[npt];
        int* xnumb = new int[npt];
        int* xvec = new int[npt];
        // -----------------------------

        for(int i=imin; i < imax + 1; i++){
            for(int j=jmin[i]; j < jmax[i]; j++){
                maps[(i * j) + j] = submap;
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

        // save_array_to_file("mapsi.dat", mapsi, ARRAY_LENGTH * npt);

        delete[] xp_segment;
        delete[] xlog;
        delete[] xnumb;
        delete[] xvec;

        return isout;
    }

    int first_map(int jmin[],     //In
                  int jmax[],     //In
                  int maps[],     //In/Out
                  int mapsi[],    //In/Out
                  int mapsw[],    //In/Out
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
    }//firstmap
}