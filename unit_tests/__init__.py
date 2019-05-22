import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
if os.getcwd() != this_dir:
    os.chdir(this_dir)

search_directory = os.path.split(this_dir)[0]
search_directory += "/tomo"
if search_directory not in sys.path:
    sys.path.append(search_directory)
