import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

# Some general functions needed for multiple examples.

# Creating new directory, or clearing existing
def make_or_clear_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        files = os.listdir(dir_name)
        for file in files:
            if os.path.isfile(f'{dir_name}/{file}'):
                os.remove(f'{dir_name}/{file}')

# Show phase-space, recreated profile and disrepancies for each iteration.
def show(image, diff, profile):
    rec_prof = np.sum(image, axis=1)

    gs = gridspec.GridSpec(2, 2)
    
    fig = plt.figure(figsize=(15, 6))
    
    img = fig.add_subplot(gs[:, 1])
    profs = fig.add_subplot(gs[0, 0])
    convg = fig.add_subplot(gs[1, 0])

    # Showing phase-space
    img.set_title('Reconstructed phase-space')
    hot_im = img.imshow(image.T, origin='lower',
                        interpolation='nearest', cmap='hot')
    divider = make_axes_locatable(img)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(hot_im, cax=cax, format='%.0e')
    

    # plotting measured and recreated profile
    profs.set_title('Reconstructed vs measured profile')
    profs.plot(profile, label='measured')
    profs.plot(rec_prof, label='reconstructed')
    profs.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    profs.legend()
    
    # plotting convergence
    convg.set_title('Distrepancy for each iteration of reconstruction')
    convg.plot(diff)
    convg.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.show()