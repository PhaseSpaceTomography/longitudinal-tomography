# General imports
import os

import matplotlib.pyplot as plt
import numpy as np

# Tomo imports
import longitudinal_tomography.utils.tomo_input as tomoin

ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', 'flatTopINDIVRotate2.dat')

# Parameters and raw data, validated on loading
tomo_data = tomoin.TomographyData.from_dat_file(in_file_pth)

tRange, ERange, density = tomo_data.run()

# %%

vmin = np.min(density[density > 0])
vmax = np.max(density)

plt.contourf(tRange * 1E9, ERange / 1E6, density.T,
             levels=np.linspace(vmin, vmax, 50), cmap='Oranges')
plt.xlabel('dt (ns)')
plt.ylabel('dE (MeV)')
plt.show()
