"""
Tomoscope specific codes

:Author(s): **Anton Lu**
"""
import logging
from os import path

import numpy as np

from ..utils import tomo_output as tomoout


log = logging.getLogger(__name__)


def save_difference(diff: np.ndarray, output_path: str, film: int):
    # Saving to file with numbers counting from one
    log.info(f'Saving saving difference to {output_path}')

    with open(path.join(output_path, f'd{film + 1:03d}.data'), 'w') as f:
        for i, d in enumerate(diff):
            if i < 10:
                f.write(f'           {i}  {d:0.7E}\n')
            else:
                f.write(f'          {i}  {d:0.7E}\n')


def save_image(xp: np.ndarray, yp: np.ndarray, weight: np.ndarray,
               n_bins: int, film: int, output_path: str):
    # Creating n_bins * n_bins phase-space image
    log.info(f'Saving picture {film}.')

    phase_space = tomoout.create_phase_space_image(xp, yp, weight, n_bins,
                                                   film)

    log.info(f'Saving image{film} to {output_path}')
    out_ps = phase_space.flatten()
    with open(path.join(output_path, f'image{film + 1:03d}.data'), 'w') as f:
        for element in out_ps:
            f.write(f'  {element:0.7E}\n')


def save_profile(prof: np.ndarray, film: int, output_path: str):

    log.info(f'Saving profile{film} to {output_path}')

    with open(path.join(output_path, f'profile{film + 1:03d}.data'), 'w') as f:
        for element in prof:
            f.write(f' {element:0.7E}\n')
