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
    gs = gridspec.GridSpec(4, 4)

    fig = plt.figure()
    
    img = fig.add_subplot(gs[1:, :3])
    profs1 = fig.add_subplot(gs[0, :3])
    profs2 = fig.add_subplot(gs[1:4, 3])
    convg = fig.add_subplot(gs[0, 3])

    cimg = img.imshow(image.T, origin='lower',
                      interpolation='nearest', cmap='hot')
    # divider = make_axes_locatable(img)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(cimg, cax=cax, format='%.0e')

    profs1.plot(np.sum(image, axis=1), label='reconstructed')
    profs1.plot(profile, label='measured')
    profs1.legend()

    profs2.plot(np.sum(image, axis=0), np.arange(image.shape[0]))

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
