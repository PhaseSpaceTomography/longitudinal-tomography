import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

wanted_working_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(wanted_working_directory)

if os.getcwd() != wanted_working_directory:
    os.chdir(wanted_working_directory)

# Making suure cpp library is compiled.
base_dir_path = os.path.realpath(__file__)[:-23]
cpp_lib_path = base_dir_path + 'tomo/cpp_files/tomolib.so'

if not os.path.isfile(cpp_lib_path):
    print('compiling...')
    compilation_script_path = base_dir_path + 'tomo/compile.py'
    subprocess.call([sys.executable, compilation_script_path])
