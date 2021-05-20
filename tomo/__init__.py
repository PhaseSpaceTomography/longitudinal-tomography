import sys

# make sure bar is in sys.modules
import longitudinal_tomography

# Or simply
sys.modules[__name__] = __import__('longitudinal_tomography')