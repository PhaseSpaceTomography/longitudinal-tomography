"""Descriptions of the input parameters, as written in JSON files, for each
format version.

:Author(s): **Simon Albright**
"""

#: Name of a parameters or frames field -> description, as written in JSON
#: files of format version 3.0.0.
PARAMETER_DESCRIPTIONS_V3_0_0 = {
    'output_dir': 'Directory in which to write all output',
    'nframes': 'Number of frames in input data',
    'skip_frames': 'Number of frames to ignore',
    'nbins_frame': 'Number of bins in each frame',
    'sampling_time': 'Width (in s) of each frame bin',
    'dturns': 'Number of machine turns between frames',
    'skip_bins_start': ('Number of frame bins before the lower profile '
                        'bound to ignore'),
    'skip_bins_end': ('Number of frame bins after the upper profile '
                      'bound to ignore'),
    'min_dt_bin': ('Number of frame bins after the lower profile bound '
                   'to treat as empty at the reconstructed time'),
    'max_dt_bin': ('Number of frame bins before the upper profile bound '
                   'to treat as empty at the reconstructed time'),
    'rebin': 'Number of frame bins to rebin into one profile bin',
    'synch_part_x': ('Time (in frame bins) from the lower profile bound '
                     'to the synchronous phase, in the bunch reference '
                     'frame (if absent or <0, a fit is performed)'),
    'demax': 'Max energy (in eV) of reconstructed phase space (if >0)',
    'filmstart': ('Index (from 0) of the first profile at which to '
                  'reconstruct'),
    'filmstop': ('Index (from 0) of the last profile at which to '
                 'reconstruct'),
    'filmstep': 'Step between reconstructions',
    'niter': 'Number of iterations for each reconstruction',
    'snpt': ('Square root of the number of test particles to track per '
             'cell'),
    'full_pp_flag': ('Flag to extend the region in phase space of map '
                     'elements (if true)'),
    'beam_ref_frame': ('Index (from 0) of the reference frame for bunch '
                       'parameters (synchronous phase, baseline, '
                       'integral)'),
    'machine_ref_frame': ('Index (from 0) of the reference frame for '
                          'machine parameters (RF voltages, B-field)'),
    'vrf1': 'Peak RF voltage (in V) of principal RF system',
    'vrf1dot': 'Time derivative of vrf1 (in V/s)',
    'vrf2': 'Peak RF voltage (in V) of higher-harmonic RF system',
    'vrf2dot': 'Time derivative of vrf2 (in V/s)',
    'h_num': 'Harmonic number of principal RF system',
    'h_ratio': 'Ratio of harmonics between RF systems',
    'phi12': ('Phase difference (in radians of the principal harmonic) '
              'between RF systems'),
    'b0': 'Dipole magnetic field (in T)',
    'bdot': 'Time derivative of b0 (in T/s)',
    'mean_orbit_rad': 'Machine radius (in m)',
    'bending_rad': 'Bending radius (in m)',
    'trans_gamma': 'Gamma transition',
    'rest_energy': 'Rest mass (in eV/c**2) of accelerated particle',
    'charge': 'Charge state of accelerated particle',
    'self_field_flag': ('Flag to include self-fields in the tracking '
                        '(if true)'),
    'g_coupling': 'Geometrical coupling coefficient',
    'zwall_over_n': ('Reactive impedance (in Ohms per mode number) over '
                     'a machine turn'),
    'pickup_sensitivity': ('Effective pick-up sensitivity (in digitizer '
                           'units per instantaneous Amp)'),
}
