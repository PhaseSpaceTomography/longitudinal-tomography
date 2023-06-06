from enum import Enum

class Mode(Enum):
    PURE = 1 # numpy
    JIT = 2
    JIT_PARALLEL = 3
    UNROLLED = 4
    UNROLLED_PARALLEL = 5
    VECTORIZE = 6 # wrong results
    VECTORIZE_PARALLEL = 7
    CPP_WRAPPER = 8
    CUPY = 9
    CUDA = 10
    CPP = 11