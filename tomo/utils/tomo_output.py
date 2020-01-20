'''Module containing functions for handling output from tomography programs.

Every function ending on 'ftn' creates an output equal to the one generated
in the Fortran program.

:Author(s): **Christoffer Hjertø Grindheim**
'''

import numpy as np
import logging as log
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# --------------------------------------------------------------- #
#                           GENERAL                               #
# --------------------------------------------------------------- #

def adjust_outpath(output_path):
    '''Functions which assurs that the given uotput path ends
    on a slash.
    
    Parameters
    ----------
    output_path: string
        Path to output directory.
    
    Returns
    -------
    output_path: string
        Path to output directory ending on a slash.
    '''
    if output_path[-1] != '/':
        output_path += '/'
    return output_path

# --------------------------------------------------------------- #
#                           PROFILES                              #
# --------------------------------------------------------------- #

def save_profile_ftn(profiles, rec_prof, output_dir):
    '''Write phase-space image to text-file in tomoscope format.
    
    Parameters
    ----------
    profiles: ndarray
        Profile measurements as a 2D array with the shape: (nprofiles, nbins).
    rec_prof: int
        Index of profile to be saved.
    output_dir: string
        Path to output directory.
    '''
    out_profile = profiles[rec_prof].flatten()
    file_path = f'{output_dir}profile{rec_prof + 1:03d}.data'
    with open(file_path, 'w') as f:
        for element in out_profile:    
            f.write(f' {element:0.7E}\n')


def save_self_volt_profile_ftn(self_fields, output_dir):
    '''Write self volts to text file in tomoscope format.

    Parameters
    ----------
    self_fields: ndarray
        Calculated self-field voltages.
    output_dir: string
        Path to output directory.
    '''
    out_profile = self_fields.flatten()
    file_path = f'{output_dir}vself.data'
    with open(file_path, 'w') as f:
        for element in out_profile:    
            f.write(f' {element:0.7E}\n')

# --------------------------------------------------------------- #
#                         PHASE-SPACE                             #
# --------------------------------------------------------------- #

def save_phase_space_ftn(image, rec_prof, output_path):
    '''Save phase-space image in a tomoscope format.

    Parameters
    ----------
    image: ndarray
        Recreated phase-space.
    rec_prof: int
        Index of reconstructed profile.
    output_dir: string
        Path to output directory.
    '''
    log.info(f'Saving image{rec_prof} to {output_path}')
    image = image.flatten()
    with open(f'{output_path}image{rec_prof + 1:03d}.data', 'w') as f:
        for element in image:
            f.write(f'  {element:0.7E}\n')

def create_phase_space_image(xp, yp, weight, n_bins, rec_prof):
    '''Convert from weighted particles to phase-space image.

    Output is equal to the phase space image created by the Fortran version. 
    
    Parameters
    ----------
    xp: ndarray
        Array containing the x coordinates of every
        particle at every time frame. Must be given in coordinates
        of the phase space coordinate system as integers. 
    yp: ndarray
        Array containing the y coordinates of every
        particle at every time frame. Must be given in coordinates
        of the phase space coordinate system as integers.
    weight: ndarray
        Array containing the weight of each particle.
    n_bins: int
        Number of bins in a profile measurment.
    rec_prof: int
        Index of reconstructed profile.
    
    Returns
    -------
    phase_space: ndarray
        Phase space presented as nbins x nbins image. Has the same
        format as the image produced by the Fortran version.

    '''
    phase_space = np.zeros((n_bins, n_bins))
    
    # Creating n_bins x n_bins phase-space image
    for x, y, w in zip(xp[:, rec_prof], yp[:, rec_prof], weight):
        phase_space[x, y] += w
    
    # Removing (if any) negative areas.
    phase_space = phase_space.clip(0.0)
    # Normalizing phase space.
    phase_space /= np.sum(phase_space)
    return phase_space

# --------------------------------------------------------------- #
#                          PLOT INFO                              #
# --------------------------------------------------------------- #

def write_plotinfo_ftn(machine, particles, profile_charge):
    '''Creates string containing plot info needed for tomoscope application.
    
    Parameters
    ----------
    machine: Machine
        Object containing machine parameters.
    particles: Particles
        Object containing particle distribution and information about
        the phase space reconstruction.
    profile_charge: float
        Total charge of a reference profile.
    
    Returns
    -------
    plot_info: string
        String containing information needed by the tomoscope application.
        The returned string has the same format as in the Fortran version.

    '''
    rec_prof = machine.filmstart
    rec_turn = rec_prof * machine.dturns

    # Check if a Fortran styled fit has been performed.
    fit_performed = True
    fit_info_vars = ['fitted_synch_part_x', 'bunchlimit_low', 'bunchlimit_up']
    for var in fit_info_vars:
        if not hasattr(machine, var):
            fit_performed = False
            break
    
    if fit_performed:
        bunchlimit_low = machine.bunchlimit_low
        bunchlimit_up = machine.bunchlimit_up
        fitted_synch_part_x = machine.fitted_synch_part_x
    else:
        bunchlimit_low = 0.0
        bunchlimit_up = 0.0
        fitted_synch_part_x = 0.0


    if particles.dEbin is None:
        raise AssertionError('dEbin has not been calculated for this '
                             'phase space info object.\n'
                             'Cannot print plot info.')
    if particles.imin is None or particles.imax is None:
        raise AssertionError('The limits in phase (I) has not been found '
                             'for this phase space info object.\n'
                             'Cannot print plot info.')  

    # '+ 1': Converting from Python to Fortran indexing
    out_s = f' plotinfo.data\n'\
            f'Number of profiles used in each reconstruction,\n'\
              f' profilecount = {machine.nprofiles}\n'\
            f'Width (in pixels) of each image = '\
              f'length (in bins) of each profile,\n'\
            f' profilelength = {machine.nbins}\n'\
            f'Width (in s) of each pixel = width of each profile bin,\n'\
            f' dtbin = {machine.dtbin:0.4E}\n'\
            f'Height (in eV) of each pixel,\n'\
            f' dEbin = {particles.dEbin:0.4E}\n'\
            f'Number of elementary charges in each image,\n'\
              f' eperimage = '\
              f'{profile_charge:0.3E}\n'\
            f'Position (in pixels) of the reference synchronous point:\n'\
            f' xat0 =  {machine.synch_part_x:.3f}\n'\
            f' yat0 =  {machine.synch_part_y:.3f}\n'\
            f'Foot tangent fit results (in bins):\n'\
            f' tangentfootl =    {bunchlimit_low:.3f}\n'\
            f' tangentfootu =    {bunchlimit_up:.3f}\n'\
            f' fit xat0 =   {fitted_synch_part_x:.3f}\n'\
            f'Synchronous phase (in radians):\n'\
            f' phi0( {rec_prof+1}) = {machine.phi0[rec_turn]:.4f}\n'\
            f'Horizontal range (in pixels) of the region in '\
              f'phase space of map elements:\n'\
            f' imin( {rec_prof+1}) =   {particles.imin} and '\
            f'imax( {rec_prof+1}) =  {particles.imax}'
    return out_s

# --------------------------------------------------------------- #
#                         DISCREPANCY                             #
# --------------------------------------------------------------- #

def save_difference_ftn(diff, output_path, rec_prof):
    '''Write reconstruction discrepancy to text file
    with tomoscope format.

    Parameters
    ----------
    diff: ndarray
        Array containing the discrepancy for the reconstructed
        phase space at each iteration.
    output_dir: string
        Path to output directory.
    rec_prof: int
        Index of profile to be saved.
    '''
    log.info(f'Saving saving difference to {output_path}')
    with open(f'{output_path}d{rec_prof + 1:03d}.data', 'w') as f:
        for i, d in enumerate(diff):
            f.write(f'         {i:3d}  {d:0.7E}\n')

# --------------------------------------------------------------- #
#                          TRACKING                               #
# --------------------------------------------------------------- #

def print_tracking_status_ftn(ref_prof, to_profile):
    '''Write output for particle tracking in Fortran style.
    The Fortran algorithm is a little different, so
    the **output concerning lost particles is not valid**.
    Meanwhile, it is needed for the tomoscope.
    Profile numbers are added by one in order to compensate for
    differences in python and fortran arrays. Fortran counts from
    one, python counts from 0.
    This function is needed by the tomography tracking algorithm.

    Parameters
    ----------
    rec_prof: int
        Index of profile to be reconstructed.
    to_profile: int
        Profile the algorithm is currently tracking towards.
    '''
    print(f' Tracking from time slice  {ref_prof + 1} to  '\
          f'{to_profile + 1},   0.000% went outside the image width.')

# --------------------------------------------------------------- #
#                         END PRODUCT                             #
# --------------------------------------------------------------- #

def show(image, diff, rec_profile):
    '''Nice presentation of reconstruction.
    
    Parameters
    ----------
    Image: ndarray
        Recreated phase-space image.
    Diff: ndarray
        Array conatining discrepancies for each iteration of reconstruction.
    Profile: ndarray
        The measured profile to be reconstructed
    '''

    # Normalizing rec_profile:
    rec_profile[:] /= np.sum(rec_profile) 

    # Creating plot
    gs = gridspec.GridSpec(4, 4)

    fig = plt.figure()
    
    img = fig.add_subplot(gs[1:, :3])
    profs1 = fig.add_subplot(gs[0, :3])
    profs2 = fig.add_subplot(gs[1:4, 3])
    convg = fig.add_subplot(gs[0, 3])

    cimg = img.imshow(image.T, origin='lower',
                      interpolation='nearest', cmap='hot')

    profs1.plot(rec_profile, label='measured')
    profs1.plot(np.sum(image, axis=1),
                label='reconstructed')
    profs1.legend()

    profs2.plot(np.sum(image, axis=0),
                np.arange(image.shape[0]))

    convg.plot(diff, label='discrepancy')
    convg.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    convg.legend()

    for ax in (profs1, profs2, convg):
        ax.set_xticks([])
        ax.set_yticks([])

    convg.set_xticks(np.arange(len(diff)))
    convg.set_xticklabels([])

    plt.gcf().set_size_inches(8, 8)
    plt.tight_layout()
    plt.show()
