from enum import Enum

class Mode(Enum):
    """Enumerator class to distinguish between the different execution modes for the application.

    Values can be either
    * CPP for the libtomo c++ implementation
    * CUDA for the GPU implementation using CUDA kernels
    * CUPY for the GPU implementation using only CuPy functions
    """
    CPP = 1
    CUDA = 2
    CUPY = 3