import sys
import os

search_directory = os.path.dirname(os.path.realpath(__file__))
if search_directory not in sys.path:
    sys.path.append(search_directory)
