import logging
from Physics import *
from Numeric import *
from utils.assertions import TomoAssertions as ta


class Parameters:

    def __init__(self):

        # Parameters to be collected from file:
        # -------------------------------------
        self.xat0 = 0.0				# Synchronous phase in bins in beam_ref_frame
        # Read form file as time (in frame bins) from the lower profile bound to the synchronous phase
        # (if < 0, a fit is performed) in the "bunch reference" frame
        self.yat0 = 0.0				# Synchronous energy (0 in relative terms) in reconstructed phase space coordinate system

        self.rebin = 0				# Rebinning factor - Number of frame bins to rebin into one profile bin
        self.rawdata_file = ""		# Input data file
        self.output_dir = ""		# Directory in which to write all output
        self.framecount = 0			# Number of frames in input data
        self.framelength = 0		    # Length of each trace in the 'raw' input file - how many bins in each frame
        self.dtbin = 0.0			# Pixel width in seconds
        self.demax = 0.0			# maximum energy of reconstructed phase space
        self.dturns = 0				# Number of machine turns between each measurement
        self.preskip_length = 0     # Subtract this number of bins from the beginning and end of the 'raw' input traces
        self.postskip_length = 0

        self.imin_skip = 0          # Number of frame bins after the lower and upper profile bound
        self.imax_skip = 0          # to treat as empty at the reconstructed time
        self.framecount = 0         # Number of frames in input data
        self.frame_skipcount = 0    # Number of frames to ignore / skip this number of traces from the beginning of the 'raw' input file
        self.framelength = 0        # Number of bins in each frame
        self.snpt = 0			    # NumPt is the number of test particles tracked from each pixel of reconstructed phase space
        self.num_iter = 0		    # Number of iterations in reconstruction process
        self.machine_ref_frame = 0  # Frame to which machine parameters are referenced (b0,VRF1,VRF2) - Frame used for gathering info about parameters
        self.beam_ref_frame = 0		# Frame to which beam parameters are referenced
        self.filmstep = 0           # step between consecutive reconstructions for the profiles from filmstart to filmstop
        self.filmstart = 0
        self.filmstop = 0
        self.full_pp_flag = False     # If set, all pixels in reconstructed phase space will be tracked

        # Machine and Particle Parameters:
        self.vrf1 = 0.0             # Peak voltage of first and second RF system at machine_ref_frame
        self.vrf2 = 0.0
        self.vrf1dot = 0.0          # Time derivatives of the RF voltages (considered constant)
        self.vrf2dot = 0.0
        self.mean_orbit_rad = 0.0   # Machine mean orbit radius (in m)
        self.bending_rad = 0.0         # Machine bending radius    (in m)
        self.b0 = 0.0               # B-field at machine_ref_frame
        self.bdot = 0.0             # Time derivative of B-field (considered constant)
        self.phi12 = 0.0            # Phase difference between the two RF systems (considered constant)
        self.h_ratio = 0.0          # Ratio of harmonics between the two RF systems
        self.h_num = 0.0            # Principle harmonic number
        self.trans_gamma = 0.0      # Transition gamma
        self.e_rest = 0.0            # Rest energy of accelerated particle
        self.q = 0.0                # Charge state of accelerated particle

        # Space charge parameters:
        self.self_field_flag = False  # Flag to include self-fields in the tracking
        self.g_coupling = 0.0         # Space charge coupling coefficient (geometrical coupling coeff.)
        self.zwall_over_n = 0.0       # Magnitude of Zwall/n, reactive impedance (in Ohms per mode number) over a machine turn
        self.pickup_sensitivity = 0.0  # Effective pick-up sensitivity (in digitizer units per instantaneous Amp)

        # calculated parameters:
        # -----------------------
        self.time_at_turn = []        # Time at each turn relative to machine_ref_frame
        self.omega_rev0 = []          # Revolution frequency at each turn
        self.phi0 = []              # Synchronous phase angle at each turn
        self.c1 = []                # TODO: Find out
        self.deltaE0 = []
        self.sfc = []               # self-field_coefficient
        self.beta0 = []             # Lorenz beta factor (v/c) at each turn
        self.eta0 = []              # Phase slip factor at each turn
        self.e0 = []                # Total energy of synchronous particle at each turn

        self.profile_count = 0       # Number of profiles
        self.profile_length = 0      # Length of profile in bins

        self.profile_mini = 0         # Index of first and last "active" index in profile
        self.profile_maxi = 0

        self.all_data = 0            # total number of data points in the 'raw' input file

        # Beam reference profile parameters (timeSpaceOutput):
        self.bunch_phaselength = 0.0  # Bunch phase length in beam reference profile
        self.tangentfoot_low = 0.0
        self.tangentfoot_up = 0.0
        self.phiwrap = 0.0
        self.wrap_length = 0
        self.fit_xat0 = 0.0
        self.x_origin = 0.0  # absolute difference in bins between phase=0 and
                             # origin of  the reconstructed phase space coordinate system.

    # Calculates parameters based on text file as input
    def get_parameters_txt(self, file_name):
        self._read_txt_input(file_name)
        self._assert_input()
        self._init_parameters()
        self._assert_parameters()

    # For retrieving parameters from text-file.
    # To be replaced by another method.
    def _read_txt_input(self, file_name):
        skiplines_start = 12

        file = open(file_name, "r")
        if file.mode == "r":
            i = 0
            while i < skiplines_start:
                file.readline()
                i += 1

            self.rawdata_file = file.readline()
            if self.rawdata_file[len(self.rawdata_file) - 1] is "\n":
                self.rawdata_file = self.rawdata_file[0:-1]

            file.readline()
            self.output_dir = file.readline()
            if self.output_dir[len(self.output_dir) - 1] is "\n":
                self.output_dir = self.output_dir[0:-1]

            file.readline()
            self.framecount = int(file.readline())

            file.readline()
            self.frame_skipcount = int(file.readline())

            file.readline()
            self.framelength = int(file.readline())

            file.readline()
            self.dtbin = float(file.readline())

            file.readline()
            self.dturns = int(file.readline())

            file.readline()
            self.preskip_length = int(file.readline())

            file.readline()
            self.postskip_length = int(file.readline())

            file.readline()
            file.readline()
            self.imin_skip = int(file.readline())

            file.readline()
            file.readline()
            self.imax_skip = int(file.readline())

            file.readline()
            self.rebin = int(file.readline())

            file.readline()
            file.readline()
            self.xat0 = float(file.readline())

            file.readline()
            self.demax = float(file.readline())

            file.readline()
            self.filmstart = int(file.readline())

            file.readline()
            self.filmstop = int(file.readline())

            file.readline()
            self.filmstep = int(file.readline())

            file.readline()
            self.num_iter = int(file.readline())

            file.readline()
            self.snpt = int(file.readline())

            file.readline()
            self.full_pp_flag = bool(int(file.readline()))

            file.readline()
            self.beam_ref_frame = int(file.readline())

            file.readline()
            self.machine_ref_frame = int(file.readline())

            file.readline()
            file.readline()
            file.readline()
            self.vrf1 = float(file.readline())

            file.readline()
            self.vrf1dot = float(file.readline())

            file.readline()
            self.vrf2 = float(file.readline())

            file.readline()
            self.vrf2dot = float(file.readline())

            file.readline()
            self.h_num = float(file.readline())

            file.readline()
            self.h_ratio = float(file.readline())

            file.readline()
            self.phi12 = float(file.readline())

            file.readline()
            self.b0 = float(file.readline())

            file.readline()
            self.bdot = float(file.readline())

            file.readline()
            self.mean_orbit_rad = float(file.readline())

            file.readline()
            self.bending_rad = float(file.readline())

            file.readline()
            self.trans_gamma = float(file.readline())

            file.readline()
            self.e_rest = float(file.readline())

            file.readline()
            self.q = float(file.readline())

            file.readline()
            file.readline()
            file.readline()
            self.self_field_flag = bool(int(file.readline()))

            file.readline()
            self.g_coupling = float(file.readline())

            file.readline()
            self.zwall_over_n = float(file.readline())

            file.readline()
            self.pickup_sensitivity = float(file.readline())

            file.close()

            logging.info("Read successful from file: " + file_name)
        else:
            raise AssertionError("Could not open file: " + file_name)

    # Subroutine for setting up parameters based on given input
    def _init_parameters(self):

        # Calculate values for each turn for arrays:
        #     time_at_turn, e0, beta0, phi0, eta0, c1, omega_rev0
        self._calc_parameter_arrays()

        self.dtbin = self.dtbin * self.rebin
        self.xat0 = self.xat0 / float(self.rebin)
        self.profile_count = self.framecount - self.frame_skipcount

        self.profile_length = (self.framelength - self.preskip_length
                               - self.postskip_length)

        # Finding min and max index in profiles.
        #   The indexes outside of these are treated as 0
        self.profile_mini, self.profile_maxi = self._find_imin_imax()

        # Total number of data points in the 'raw' input file
        self.all_data = self.framecount * self.framelength

        # Find self field coefficient for each profile
        self.sfc = calc_self_field_coeffs(self)

    # Fills up arrays with zeroes, ready to use further.
    # + 1 to both include turn#0 and the very last turn.
    def _init_arrays(self, all_turns):
        array_length = all_turns + 1
        self.time_at_turn = np.zeros(array_length)
        self.omega_rev0 = np.zeros(array_length)
        self.phi0 = np.zeros(array_length)
        self.c1 = np.zeros(array_length)
        self.deltaE0 = np.zeros(array_length)
        self.beta0 = np.zeros(array_length)
        self.eta0 = np.zeros(array_length)
        self.e0 = np.zeros(array_length)

    def _array_initial_values(self):
        i0 = (self.machine_ref_frame - 1) * self.dturns
        self.time_at_turn[i0] = 0
        self.e0[i0] = b_to_e(self)
        self.beta0[i0] = lorenz_beta(self, i0)
        phi_lower, phi_upper = find_phi_lower_upper(self, i0)
        self.phi0[i0] = find_synch_phase(self, i0, phi_lower, phi_upper)
        return i0

    # Filling up the rest of the parameter arrays:
    #   time_at_turn, e0, beta0, phi0, deltaE0
    def _calc_parameter_arrays(self):
        all_turns = self._calc_number_of_turns()
        self._init_arrays(all_turns)
        i0 = self._array_initial_values()

        for i in range(i0 + 1, all_turns + 1):
            self.time_at_turn[i] = (self.time_at_turn[i - 1]
                                    + 2*np.pi*self.mean_orbit_rad
                                    / (self.beta0[i - 1]*C))

            self.phi0[i] = newton(rf_voltage, drf_voltage,
                                  self.phi0[i - 1], self, i, 0.001)

            self.e0[i] = (self.e0[i - 1]
                          + self.q
                          * short_rf_voltage_formula(
                                self.phi0[i], self.vrf1, self.vrf1dot,
                                self.vrf2, self.vrf2dot, self.h_ratio,
                                self.phi12, self.time_at_turn, i))

            self.beta0[i] = np.sqrt(1.0 - (self.e_rest/float(self.e0[i]))**2)
            self.deltaE0[i] = self.e0[i] - self.e0[i - 1]
        for i in range(i0 - 1, 0, -1):
            self.e0[i] = (self.e0[i + i]
                          - self.q
                          * short_rf_voltage_formula(self.phi0[i], self, i))
            self.beta0[i] = np.sqrt(1.0 * -(self.e_rest/self.e0[i])**2)
            self.deltaE0[i] = self.e0[i + 1] - self.e0[i]
            self.time_at_turn[i] = (self.time_at_turn[i + 1]
                                    - 2 * np.pi * self.mean_orbit_rad
                                    / (self.beta0[i] * C))
            self.phi0[i] = newton(rf_voltage, drf_voltage,
                                  self.phi0[i + 1], self, i, 0.001)

        # Calculate phase slip factor at each turn
        self.eta0 = phase_slip_factor(self)

        # Calculates c1 for each turn
        # TODO: what is c1?
        self.c1 = find_c1(self)

        # Calculate revolution frequency at each turn
        self.omega_rev0 = revolution_freq(self)

    # Find profile_mini and profile_maxi
    def _find_imin_imax(self):
        profile_mini = self.imin_skip/self.rebin
        if (self.profile_length - self.imax_skip) % self.rebin == 0:
            profile_maxi = ((self.profile_length - self.imax_skip)
                            / self.rebin)
        else:
            profile_maxi = ((self.profile_length - self.imax_skip)
                            / self.rebin + 1)
        return int(profile_mini), int(profile_maxi)

    def _assert_input(self):
        # Note that some of the assertions is setting the lower limit as 1.
        # This is because of calibrating from input files meant for Fortran,
        #    where arrays by default starts from 1, to the Python version
        #    with arrays starting from 0.

        # Frame assertions
        ta.assert_greater(self.framecount, "frame count", 0, InputError)
        ta.assert_inrange(self.frame_skipcount, "frame skip-count",
                          0, self.framecount, InputError)
        ta.assert_greater(self.framelength, "frame length", 0, InputError)
        ta.assert_inrange(self.preskip_length, "pre-skip length",
                          0, self.framelength, InputError)
        ta.assert_inrange(self.postskip_length, "post-skip length",
                          0, self.framelength, InputError)

        # Bin assertions
        ta.assert_greater(self.dtbin, "dtbin", 0, InputError,
                          'NB: dtbin is the difference of time in bin')
        ta.assert_greater(self.dturns, "dturns", 0, InputError,
                          'NB: dturns is the number of machine turns'
                          'between each measurement')
        ta.assert_inrange(self.imin_skip, 'imin skip',
                          0, self.framelength, InputError)
        ta.assert_inrange(self.imax_skip, 'imax skip',
                          0, self.framelength, InputError)
        ta.assert_greater_or_equal(self.rebin, 're-binning factor',
                                   1, InputError)

        # Assertions: profile to be reconstructed
        ta.assert_greater_or_equal(self.filmstart, 'film start',
                                   1, InputError)
        ta.assert_greater_or_equal(self.filmstop, 'film stop',
                                   self.filmstart, InputError)
        ta.assert_less_or_equal(abs(self.filmstep), 'film step',
                                abs(self.filmstop - self.filmstart + 1),
                                InputError)
        ta.assert_not_equal(self.filmstep, 'film step', 0, InputError)

        # Reconstruction parameter assertions
        ta.assert_greater(self.num_iter, 'num_iter', 0, InputError,
                          'NB: num_iter is the number of iterations of the '
                          'reconstruction process')
        ta.assert_greater(self.snpt, 'snpt', 0, InputError,
                          'NB: snpt is the square root '
                          'of #tracked particles.')

        # Reference frame assertions
        ta.assert_greater_or_equal(self.machine_ref_frame,
                                   'machine ref. frame',
                                   1, InputError)
        ta.assert_greater_or_equal(self.beam_ref_frame, 'beam ref. frame',
                                   1, InputError)

        # Machine parameter assertion
        ta.assert_greater_or_equal(self.h_num, 'harmonic number',
                                   1, MachineParameterError)
        ta.assert_greater_or_equal(self.h_ratio, 'harmonic ratio',
                                   1, MachineParameterError)
        ta.assert_greater(self.b0, 'B field (B0)',
                          0, MachineParameterError)
        ta.assert_greater(self.mean_orbit_rad, "mean orbit radius",
                          0, MachineParameterError)
        ta.assert_greater(self.bending_rad, "Bending radius",
                          0, MachineParameterError)
        ta.assert_greater(self.e_rest, 'rest energy',
                          0, MachineParameterError)

        # Space charge parameter assertion
        ta.assert_greater_or_equal(self.zwall_over_n, 'z wall over n',
                                   0, SpaceChargeParameterError)
        ta.assert_greater_or_equal(self.pickup_sensitivity,
                                   'pick-up sensitivity',
                                   0, SpaceChargeParameterError)
        ta.assert_greater_or_equal(self.g_coupling, 'g_coupling',
                                   0, SpaceChargeParameterError,
                                   'NB: g_coupling:'
                                   'geometrical coupling coefficient')

    def _assert_parameters(self):
        # Calculated parameters
        ta.assert_greater_or_equal(self.profile_length, 'profile length', 0,
                                   InputError,
                                   f'Make sure that the sum of post- and'
                                   f'pre-skip length is less'
                                   f'than the frame length\n'
                                   f'frame length: {self.framelength}\n'
                                   f'pre-skip length: {self.preskip_length}\n'
                                   f'post-skip length: {self.postskip_length}')

        ta.assert_array_shape_equal([self.time_at_turn,
                                     self.omega_rev0,
                                     self.phi0,
                                     self.c1,
                                     self.deltaE0,
                                     self.beta0,
                                     self.eta0,
                                     self.e0],
                                    ['time_at_turn',
                                     'omega_re0',
                                     'phi0',
                                     'c1',
                                     'deltaE0',
                                     'beta0',
                                     'eta0',
                                     'e0'],
                                    (self._calc_number_of_turns() + 1, ))

    def _calc_number_of_turns(self):
        all_turns = (self.framecount - self.frame_skipcount - 1) * self.dturns
        ta.assert_greater(all_turns, 'all_turns', 0, InputError,
                          'Make sure that frame skip-count'
                          'do not exceed number of frames')
        return all_turns
