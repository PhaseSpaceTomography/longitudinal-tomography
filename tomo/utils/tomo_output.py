import numpy as np
import logging as log
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# --------------------------------------------------------------- #
#                           GENERAL                               #
# --------------------------------------------------------------- #

# Make sure that output path ends on a dash.
def adjust_outpath(output_path):
    if output_path[-1] != '/':
        output_path += '/'
    return output_path

# --------------------------------------------------------------- #
#                           PROFILES                              #
# --------------------------------------------------------------- #

# Write phase-space image to text-file in tomoscope format.
# Profiles: waterfall of profiles.
# rec_prof: index of profile to be reconstructed.
def save_profile_ftn(profiles, rec_prof, output_dir):
    out_profile = profiles[rec_prof].flatten()
    file_path = f'{output_dir}profile{rec_prof + 1:03d}.data'
    with open(file_path, 'w') as f:
        for element in out_profile:    
            f.write(f' {element:0.7E}\n')


# Write self volts to text file in tomoscope format.
# self_fields: calculated self field voltages.
def save_self_volt_profile_ftn(self_fields, output_dir):
    out_profile = self_fields.flatten()
    file_path = f'{output_dir}vself.data'
    with open(file_path, 'w') as f:
        for element in out_profile:    
            f.write(f' {element:0.7E}\n')

# --------------------------------------------------------------- #
#                         PHASE-SPACE                             #
# --------------------------------------------------------------- #

# Save phase-space image in a tomoscope format
# image: recreated phase-space
# rec prof: index of profile to be reconstructed. 
def save_phase_space_ftn(image, rec_prof, output_path):
    log.info(f'Saving image{rec_prof} to {output_path}')
    image = image.flatten()
    with open(f'{output_path}image{rec_prof + 1:03d}.data', 'w') as f:
        for element in image:
            f.write(f'  {element:0.7E}\n')

# Convert from weighted particles to phase-space image.
def create_phase_space_image(xp, yp, weight, n_bins, rec_prof):
    phase_space = np.zeros((n_bins, n_bins))
    
    # Creating n_bins * n_bins phase-space image
    for x, y, w in zip(xp[:, rec_prof], yp[:, rec_prof], weight):
        phase_space[x, y] += w
    
    phase_space = phase_space.clip(0.0)
    phase_space /= np.sum(phase_space)
    return phase_space

# --------------------------------------------------------------- #
#                          PLOT INFO                              #
# --------------------------------------------------------------- #

# Returns string containing plot info for tomoscope application
# '+ 1': Converting from Python to Fortran indexing
def write_plotinfo_ftn(ps_info, profile_charge):
    rec_prof = ps_info.machine.filmstart
    rec_turn = rec_prof * ps_info.machine.dturns

    # Check if a Fortran styled fit has been performed.
    fit_performed = True
    fit_info_vars = ['fitted_xat0', 'tangentfoot_low', 'tangentfoot_up']
    for var in fit_info_vars:
        if not hasattr(ps_info.machine, var):
            fit_performed = False
            break
    if fit_performed:
        tangentfoot_low = ps_info.machine.tangentfoot_low
        tangentfoot_up = ps_info.machine.tangentfoot_up
        fitted_xat0 = ps_info.machine.fitted_xat0
    else:
        tangentfoot_low = 0.0
        tangentfoot_up = 0.0
        fitted_xat0 = 0.0

    out_s = f' plotinfo.data\n'\
            f'Number of profiles used in each reconstruction,\n'\
              f' profilecount = {ps_info.machine.nprofiles}\n'\
            f'Width (in pixels) of each image = '\
              f'length (in bins) of each profile,\n'\
            f' profilelength = {ps_info.machine.nbins}\n'\
            f'Width (in s) of each pixel = width of each profile bin,\n'\
            f' dtbin = {ps_info.machine.dtbin:0.4E}\n'\
            f'Height (in eV) of each pixel,\n'\
            f' dEbin = {ps_info.dEbin:0.4E}\n'\
            f'Number of elementary charges in each image,\n'\
              f' eperimage = '\
              f'{profile_charge:0.3E}\n'\
            f'Position (in pixels) of the reference synchronous point:\n'\
            f' xat0 =  {ps_info.machine.xat0:.3f}\n'\
            f' yat0 =  {ps_info.machine.yat0:.3f}\n'\
            f'Foot tangent fit results (in bins):\n'\
            f' tangentfootl =    {tangentfoot_low:.3f}\n'\
            f' tangentfootu =    {tangentfoot_up:.3f}\n'\
            f' fit xat0 =   {fitted_xat0:.3f}\n'\
            f'Synchronous phase (in radians):\n'\
            f' phi0( {rec_prof+1}) = {ps_info.machine.phi0[rec_turn]:.4f}\n'\
            f'Horizontal range (in pixels) of the region in '\
              f'phase space of map elements:\n'\
            f' imin( {rec_prof+1}) =   {ps_info.imin} and '\
            f'imax( {rec_prof+1}) =  {ps_info.imax}'
    return out_s

# --------------------------------------------------------------- #
#                         DISCREPANCY                             #
# --------------------------------------------------------------- #

# Write difference to text file in tomoscope format
def save_difference_ftn(diff, output_path, rec_prof):
    log.info(f'Saving saving difference to {output_path}')
    with open(f'{output_path}d{rec_prof + 1:03d}.data', 'w') as f:
        for i, d in enumerate(diff):
            f.write(f'         {i:3d}  {d:0.7E}\n')

# --------------------------------------------------------------- #
#                          TRACKING                               #
# --------------------------------------------------------------- #

# Write output for particle tracking in Fortran style.
# The Fortran algorithm is a little different, so
#  the output concerning lost particles is not valid.
#  Meanwhile, it is needed for the tomoscope.
# Profile numbers are added by one in order to compensate for
#  differences in python and fortran arrays. Fortrans counts from
#  one, python counts from 0.
def print_tracking_status_ftn(ref_prof, to_profile):
    print(f' Tracking from time slice  {ref_prof + 1} to  '\
          f'{to_profile + 1},   0.000% went outside the image width.')

# --------------------------------------------------------------- #
#                         END PRODUCT                             #
# --------------------------------------------------------------- #

# Nice plot of output.
# Image: recreated phase-space image
# Diff: array conatining discrepancies for each iteration of reconstruction.
# Profile: The measured profile to be reconstructed
def show(image, diff, rec_profile):

    # Normalizing rec_profile:
    rec_profile[:] /= np.sum(rec_profile) 

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
