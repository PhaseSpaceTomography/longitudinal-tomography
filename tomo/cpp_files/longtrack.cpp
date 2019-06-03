 #include <iostream>
 #include <fstream>
 #include <stdlib.h>
 #include <cmath>
 #include <vector>
 #include <string>

using namespace std;

int longtrack(double xp[],                  // inout
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
              int h_num, double h_ratio);
void save_array_to_file(string, double [], int);
double* load_doubles_from_file(int &, string);
int* load_ints_from_file(int &, string);
int lines_in_file(ifstream &);
double* calculate_dphi(const double xp[],
                      int xp_len, 
                      double xorigin, int h_num,
                      double omega_rev0, double dtbin,
                      double phi0);
double* calculate_denergy(const double yp[], int yp_len,
                          double yat0, double dEbin);

int main(){
    cout.precision(17);
    string directory = "/home/cgrindhe/tomo_v3/unit_tests/resources/C500MidPhaseNoise/";

    cout << "Loading parameters..." << endl;
    int xp_len;
    int yp_len;
    int or_len;
    int phi0_len;
    int c1_len;
    int dE0_len;
    int tt_len;
    double * xp = load_doubles_from_file(xp_len, "/home/cgrindhe/cpp_test/init_xp.dat");
    double * yp = load_doubles_from_file(yp_len, "/home/cgrindhe/cpp_test/init_yp.dat");
    double * omega_rev0 = load_doubles_from_file(or_len, directory + "omegarev0.dat");
    double * phi0 = load_doubles_from_file(phi0_len,  directory + "phi0.dat");
    double * c1 = load_doubles_from_file(c1_len, directory + "c1.dat");
    double * turn_time = load_doubles_from_file(dE0_len, directory + "time_at_turn.dat");
    double * deltaE0 = load_doubles_from_file(tt_len, directory + "deltaE0.dat");
    double xorigin = -69.73326295579088;
    int h_num = 1;
    double dtbin = 1.9999999999999997e-09;
    double yat0 = 102.5;
    double dEbin = 23340.63328895732;
    int turn_now = 0;
    int nreps = 12;
    double q = 1.0;
    double phi12 = 0.3116495273194016;
    int direction = 1;
    double vrf1 = 7945.403672852664;
    double vrf1dot = 0.0;
    double vrf2 = -0.0;
    double vrf2dot = 0.0;
    double h_ratio = 2.0;


    cout << "Running longtrack: " << endl;
    turn_now = longtrack(xp, yp, omega_rev0, phi0, c1, turn_time,
                         deltaE0, xp_len, xorigin, dtbin, dEbin, yat0,
                         phi12, direction, nreps, turn_now, q,
                         vrf1, vrf1dot, vrf2, vrf2dot, h_num, h_ratio);

    save_array_to_file("/home/cgrindhe/temp/xp_out.dat", xp, xp_len);
    save_array_to_file("/home/cgrindhe/temp/yp_out.dat", yp, xp_len);

    delete[] xp;
    delete[] yp;
    delete[] omega_rev0;
    delete[] phi0;
    delete[] c1;
    delete[] turn_time;
    delete[] deltaE0;
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
    
    double * dphi = calculate_dphi(xp, xp_len, xorigin, h_num,
                                   omega_rev0[turn_now], dtbin,
                                   phi0[turn_now]);
    double * denergy =  calculate_denergy(yp, xp_len, yat0, dEbin);

    int i=0, j=0;
    if(direction > 0){
        for(i=0; i < nreps; i++){
            
            for(j=0; j < xp_len; j++){
                dphi[j] -= c1[turn_now] * denergy[j];
            }
            turn_now++;
            for(j=0; j < xp_len; j++){
                denergy[j] += q * (vrf1 + vrf1dot * turn_time[turn_now]) // flytt ut av for (j..)
                                   * sin(dphi[j] + phi0[turn_now])
                                   + (vrf2 + vrf2dot * turn_time[turn_now])
                                   * sin(h_ratio 
                                         * (dphi[j] + phi0[turn_now] - phi12))
                                   - deltaE0[turn_now];
            }
        }
    }
    else{
        throw new exception;
    }

    for(i=0; i < xp_len; i++){
        xp[i] = (dphi[i] + phi0[turn_now])
                / (static_cast<double>(h_num) * omega_rev0[turn_now] * dtbin)
                - xorigin;
    }
    for(i=0; i < xp_len; i++){
        yp[i] = denergy[i] / static_cast<double>(dEbin) + yat0;
    }

    delete[] dphi;
    delete[] denergy;

    return turn_now;
}

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

int* load_ints_from_file(int& size_out, string file_path){
    ifstream infile;
        
    infile.open(file_path.c_str());
    if(!infile){
        exit(EXIT_FAILURE);
    }
    size_out = lines_in_file(infile);
    infile.close();

    int* ans = new int[size_out];
          
    infile.open(file_path.c_str());
    for(int i=0; i < size_out; i++){
        infile >> ans[i];
    }
    infile.close();
    return ans;
}

double* load_doubles_from_file(int& size_out, string file_path){
    ifstream infile;
        
    infile.open(file_path.c_str());
    if(!infile){
        exit(EXIT_FAILURE);
    }
    size_out = lines_in_file(infile);
    infile.close();

    double* ans = new double[size_out];
          
    infile.open(file_path.c_str());
    for(int i=0; i < size_out; i++){
        infile >> ans[i];
    }
    infile.close();
    return ans;
}

int lines_in_file(ifstream &in_file){
    if(in_file){
        string line;
        int line_counter = 0;

        while (!in_file.eof()){
            getline(in_file, line);
            line_counter++;
        }
        return line_counter;
    }
    else
    {
        return -1;
    }
}

void save_array_to_file(string filename, double array[],
                            int array_length){
    ofstream out_file;
    out_file.open(filename.c_str());
    if(out_file){
        for(int i=0; i < array_length; i++){
            out_file << array[i] << endl;
        }
        out_file.close();
    }   
}