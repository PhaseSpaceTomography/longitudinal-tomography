# General imports
import os

import matplotlib.pyplot as plt
import numpy as np

# Tomo imports
import longitudinal_tomography.utils.tomo_run as tomorun
import longitudinal_tomography

longitudinal_tomography.use_gpu()

ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', 'flatTopINDIVRotate2.dat')

tRange, ERange, density = tomorun.run(in_file_pth)

# %%

vmin = np.min(density[density > 0])
vmax = np.max(density)

plt.contourf(tRange * 1E9, ERange / 1E6, density.T,
              levels=np.linspace(vmin, vmax, 50), cmap='Oranges')
plt.xlabel('dt (ns)')
plt.ylabel('dE (MeV)')
plt.show()
