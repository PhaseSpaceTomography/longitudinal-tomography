import logging
from os import path

import numpy as np

from ..data import data_treatment as dtreat


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
               n_bins: float, film: int, output_path: str):
    # Creating n_bins * n_bins phase-space image
    log.info(f'Saving picture {film}.')

    phase_space = dtreat._make_phase_space(xp[:, film], yp[:, film],
                                           weight, n_bins)

    # Suppressing negative numbers
    phase_space = phase_space.clip(0.0)

    # Normalizing
    phase_space /= np.sum(phase_space)

    log.info(f'Saving image{film} to {output_path}')
    out_ps = phase_space.flatten()
    with open(path.join(output_path, f'image{film + 1:03d}.data'), 'w') as f:
        for element in out_ps:
            f.write(f'  {element:0.7E}\n')
