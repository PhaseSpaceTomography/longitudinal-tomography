import os
import sys
import numpy as np
import logging as log
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.exceptions import InputError

class OutputHandler:
    
    # TO BE ADDED:
    # - saving all outut in the same way as fortran

    # --------------------------------------------------------------- #
    #                         UTILITIES                               #
    # --------------------------------------------------------------- #

    @classmethod
    # Assert that output path ends on a '/'
    def adjust_outpath(cls, output_path):
        if output_path[-1] != '/':
            output_path += '/'
        return output_path

    # --------------------------------------------------------------- #
    #                           PROFILES                              #
    # --------------------------------------------------------------- #

    @classmethod
    # Write phase space image to text-file in tomoscope format.
    def save_profile_ccc(cls, profiles, prof_idx, output_dir):
        out_profile = profiles[prof_idx].flatten()
        file_path = f'{output_dir}profile{prof_idx + 1:03d}.data'
        with open(file_path, 'w') as f:
            for element in out_profile:    
                f.write(f' {element:0.7E}\n')

    @classmethod
    def save_self_volt_profile_ccc(cls, profiles_sv, output_dir):
        out_profile = profiles_sv.flatten()
        file_path = f'{output_dir}vself.data'
        with open(file_path, 'w') as f:
            for element in out_profile:    
                f.write(f' {element:0.7E}\n')

    # --------------------------------------------------------------- #
    #                         PHASE-SPACE                             #
    # --------------------------------------------------------------- #

    # Write phase space image to .npy file.
    @classmethod
    def save_phase_space_npy(cls, phase_space, film, output_path):
        log.info(f'Saving image{film} to {output_path}')
        np.save(f'{output_path}image{film + 1:03d}', phase_space)

    # Write phase space image to text-file in tomoscope format.
    @classmethod
    def save_phase_space_ccc(cls, phase_space, film, output_path):
        log.info(f'Saving image{film} to {output_path}')
        phase_space = phase_space.flatten()
        with open(f'{output_path}image{film + 1:03d}.data', 'w') as f:
            for element in phase_space:
                f.write(f'  {element:0.7E}\n')

    @staticmethod
    def create_phase_space_image(xp, yp, weight, n_bins, film):
        phase_space = np.zeros((n_bins, n_bins))
    
        # Creating n_bins * n_bins phase-space image
        for x, y, w in zip(xp[:, film], yp[:, film], weight):
            phase_space[x, y] += w
    
        # Surpressing negative numbers
        phase_space = phase_space.clip(0.0)

        # Normalizing
        phase_space /= np.sum(phase_space)

        return phase_space

    # --------------------------------------------------------------- #
    #                         DISCREPANCY                             #
    # --------------------------------------------------------------- #
    
    # Write difference to text file in tomoscope format
    @classmethod
    def save_difference_ccc(cls, diff, output_path, film):
        # Saving to file with numbers counting from one
        log.info(f'Saving saving difference to {output_path}')
        with open(f'{output_path}d{film + 1:03d}.data', 'w') as f:
            for i, d in enumerate(diff):
                f.write(f'         {i:3d}  {d:0.7E}\n')

    # --------------------------------------------------------------- #
    #                          TRACKING                               #
    # --------------------------------------------------------------- #

    # Write output for particle tracking in Fortran style.
    # The Fortran algorithm is a little different, so
    #  the last part is not valid. Meanwhile, it is needed
    #  for the tomoscope. To be changed in future.
    # Profile numbers are added by one in order to compensate for
    # differences in python and fortran arrays. Fortrans counts from
    # one, python counts from 0. 
    @classmethod
    def print_tracking_status_ccc(cls, ref_prof, to_profile):
        print(f' Tracking from time slice  {ref_prof + 1} to  '\
              f'{to_profile + 1},   0.000% went outside the image width.')

    # --------------------------------------------------------------- #
    #                         COORDINATES                             #
    # --------------------------------------------------------------- #

    @classmethod
    def save_coordinates_npy(cls, xp, yp, output_path):
        log.info(f'Saving saving coordinates to {output_path}')
        log.info('Saving xp')
        np.save(output_path + 'xp', xp)
        log.info('Saving yp')
        np.save(output_path + 'yp', yp)

    # --------------------------------------------------------------- #
    #                         END PRODUCT                             #
    # --------------------------------------------------------------- #

    @classmethod
    def show(cls, image, diff, profile):
        gs = gridspec.GridSpec(4, 4)

        fig = plt.figure()
    
        img = fig.add_subplot(gs[1:, :3])
        profs1 = fig.add_subplot(gs[0, :3])
        profs2 = fig.add_subplot(gs[1:4, 3])
        convg = fig.add_subplot(gs[0, 3])

        cimg = img.imshow(image.T, origin='lower',
                          interpolation='nearest', cmap='hot')

        profs1.plot(profile, label='measured')
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
