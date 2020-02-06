#General imports
import numpy as np
import matplotlib.pyplot as plt
import os

#Tomo imports
import tomo.utils.tomo_run as tomorun


ex_dir = os.path.realpath(os.path.dirname(__file__)).split('/')[:-1]
in_file_pth = '/'.join(ex_dir + ['/input_files/flatTopINDIVRotate2.dat'])
tomorun.run_file(in_file_pth)