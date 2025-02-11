import subprocess
import os
from .. import exceptions as expt

REDUCTION_BLOCK_SIZE = 32

cuda_sources = [
    os.path.dirname(__file__) + "/kick_and_drift",
    os.path.dirname(__file__) + "/reconstruct",
]

def compile_kernels():
    """
    Compiles the CUDA source files to binaries which will be imported.
    If it cannot find nvcc naturally, please set the environment variable CUDA_PATH
    to be able to call the compiler.
    """

    print("Compiling the CUDA sources")

    cuda_path = os.getenv("CUDA_PATH", default="")
    if cuda_path != "":
        nvcc = cuda_path + "/bin/nvcc"
    else:
        nvcc = "nvcc"

    nvccflags = [nvcc, "--cubin", "-O3", "--use_fast_math"]

    try:
        import cupy as cp
        dev = cp.cuda.Device(0)
    except ImportError as e:
        raise expt.CudaCompilationException(
            "Package CuPy is not installed. CUDA sources cannot be compiled"
        ) from e
    except cp.cuda.runtime.CUDARuntimeError as e:
        raise expt.CudaCompilationException(
            "No capable GPU device has been found.\
                CUDA sources cannot be compiled"
        ) from e
    else:
        dev_name = cp.cuda.runtime.getDeviceProperties(dev)["name"]
        print("Discovering the device compute capability..")
        comp_capability = dev.compute_capability
        print(f"Device name {dev_name}")

        print(f"Compiling the sources for architecture {comp_capability}.")
        nvccflags += ["-arch", f"sm_{comp_capability}"]

        try:
            for source in cuda_sources:
                command = nvccflags + [
                    "-o",
                    source + "_double.cubin",
                    source + ".cu",
                    f"-DBLOCK_SIZE={REDUCTION_BLOCK_SIZE}"
                ]
                subprocess.run(command)
                command = nvccflags + [
                    "-o",
                    source + "_single.cubin",
                    source + ".cu",
                    "-DUSEFLOAT",
                    f"-DBLOCK_SIZE={REDUCTION_BLOCK_SIZE}"
                ]
                subprocess.run(command)
        except FileNotFoundError as e:
            raise expt.CudaCompilationException(
                "The NVCC compiler could not be found. "\
                + "Please check if CUDA_PATH has been set properly. "\
                + f"Subprocess returned the following error: {e}"
            )

        if check_compiled_kernels():
            print("The CUDA sources have been successfully compiled.")
        else:
            raise expt.CudaCompilationException(
                "The CUDA source compilation failed."
        )

def check_compiled_kernels():
    """
    Check if the compiled CUDA kernels for both double and single precision exist.

    Returns:
        bool: True if all compiled CUDA kernels exist, False otherwise.
    """
    return os.path.isfile(cuda_sources[0] + "_double.cubin")\
        and os.path.isfile(cuda_sources[1] + "_double.cubin")\
        and os.path.isfile(cuda_sources[0] + "_single.cubin")\
        and os.path.isfile(cuda_sources[1] + "_single.cubin")