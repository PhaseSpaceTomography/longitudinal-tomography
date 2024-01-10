import subprocess
import os
from .. import exceptions as expt

REDUCTION_BLOCK_SIZE = 32

def compile_gpu():
    """Compiles the CUDA source files to binaries which will be imported.
    If it cannot find nvcc naturally, please set the environment variable CUDA_PATH
    to be able to call the compiler.
    """
    
    cuda_sources = [
        os.path.dirname(__file__) + "/kick_and_drift",
        os.path.dirname(__file__) + "/reconstruct",
    ]

    print(os.path.dirname(__file__))

    if not os.path.isfile(cuda_sources[0] + ".cubin") or not os.path.isfile(
        cuda_sources[1] + ".cubin"
    ):
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

            for source in cuda_sources:
                command = nvccflags + ["-o", source + "_double.cubin", source + ".cu"]
                subprocess.run(command)
                command = nvccflags + [
                    "-o",
                    source + "_single.cubin",
                    source + ".cu",
                    "-DUSEFLOAT",
                    f"-DBLOCK_SIZE={REDUCTION_BLOCK_SIZE}"
                ]
                subprocess.run(command)

            if (
                os.path.isfile(cuda_sources[0] + "_double.cubin")
                and os.path.isfile(cuda_sources[1] + "_double.cubin")
                and os.path.isfile(cuda_sources[0] + "_single.cubin")
                and os.path.isfile(cuda_sources[1] + "_single.cubin")
            ):
                print("The CUDA sources have been successfully compiled.")
            else:
                raise expt.CudaCompilationException(
                    "The CUDA source compilation failed.\
                    Check if CUDA_PATH has been set."
            )
