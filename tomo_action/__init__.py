import os
import logging

logging.basicConfig(level=logging.INFO)

wanted_working_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(wanted_working_directory)

if os.getcwd() != wanted_working_directory:
    os.chdir(wanted_working_directory)
