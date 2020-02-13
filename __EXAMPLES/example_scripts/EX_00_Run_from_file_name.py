#General imports
import numpy as np
import matplotlib.pyplot as plt
import os

#Tomo imports
import tomo.utils.tomo_run as tomorun

ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', 'flatTopINDIVRotate2.dat')

tRange, ERange, density = tomorun.run_file(in_file_pth)

#%%

vmin = np.min(density[density>0])
vmax = np.max(density)

plt.contourf(tRange*1E9, ERange/1E6, density.T, 
             levels=np.linspace(vmin, vmax, 50), cmap='Oranges')
plt.xlabel('dt (ns)')
plt.ylabel('dE (MeV)')
plt.show()