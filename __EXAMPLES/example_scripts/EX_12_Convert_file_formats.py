# General imports
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

# Tomo imports
import longitudinal_tomography.utils.file_handling as fhand
import longitudinal_tomography.utils.tomo_data as tdata

ex_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
in_file_pth = os.path.join(ex_dir, 'input_files', 'flatTopINDIV8thOrder.dat')
out_dir = tempfile.mkdtemp()
json_pth = os.path.join(out_dir, 'flatTopINDIV8thOrder.json')
dat_pth = os.path.join(out_dir, 'flatTopINDIV8thOrder.dat')

# The format version of a file is found from its content
print(f'{in_file_pth} is version {fhand.detect_version(in_file_pth)}')

# .dat to JSON, the default version, then back to .dat
fhand.convert(in_file_pth, json_pth)
fhand.convert(json_pth, dat_pth, fhand.DAT_VERSION)
print(f'{json_pth} is version {fhand.detect_version(json_pth)}')
print(f'{dat_pth} is version {fhand.detect_version(dat_pth)}')

# Files of every version are read into the same TomographyData
tomo_data = tdata.TomographyData.from_file(json_pth)
print('JSON holds the same data as the .dat file:',
      tomo_data == tdata.TomographyData.from_file(in_file_pth))

tRange, ERange, density = tomo_data.run()

# %%

vmin = np.min(density[density > 0])
vmax = np.max(density)

plt.contourf(tRange * 1E9, ERange / 1E6, density.T,
             levels=np.linspace(vmin, vmax, 50), cmap='Oranges')
plt.xlabel('dt (ns)')
plt.ylabel('dE (MeV)')
plt.show()
