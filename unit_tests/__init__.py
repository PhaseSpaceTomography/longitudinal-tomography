import sys
import os

search_directory = os.path.dirname(os.path.realpath(__file__))
search_directory = os.path.split(search_directory)[0]
search_directory += "/tomo"
if search_directory not in sys.path:
    sys.path.append(search_directory)
