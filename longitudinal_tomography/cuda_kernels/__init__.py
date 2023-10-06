import subprocess
import os
from .. import exceptions as expt

cuda_sources = [
    os.path.dirname(__file__) + '/kick_and_drift.cu',
    os.path.dirname(__file__) + '/reconstruct.cu',
]

print(os.path.dirname(__file__))

if not os.path.isfile(cuda_sources[0] + 'bin') or not os.path.isfile(cuda_sources[1] + 'bin'):
    print('Compiling the CUDA sources')

    cuda_path = os.getenv('CUDA_PATH', default='')
    if cuda_path != '':
        nvcc = cuda_path + '/bin/nvcc'
    else:
        nvcc = 'nvcc'

    nvccflags = [nvcc, '--cubin', '-O3', '--use_fast_math']

    try:
        import cupy as cp
        dev = cp.cuda.Device(0)
        dev_name = cp.cuda.runtime.getDeviceProperties(dev)['name']
        print('Discovering the device compute capability..')
        comp_capability = dev.compute_capability
        print(f'Device name {dev_name}')

        print(f'Compiling the sources for architecture {comp_capability}.')
        nvccflags += ['-arch', f'sm_{comp_capability}']

        for source in cuda_sources:
            command = nvccflags + ['-o', source + 'bin', source]
            subprocess.call(command)

        if os.path.isfile(cuda_sources[0] + 'bin') and os.path.isfile(cuda_sources[1] + 'bin'):
            print('The CUDA sources have been successfully compiled.')
        else:
            raise expt.CudaCompilationException('The CUDA source compilation failed.')
    except ImportError as e:
        raise expt.CudaCompilationException("Package CuPy is not installed. CUDA sources cannot be compiled") from e
    except cp.cuda.runtime.CUDARuntimeError as e:
        raise expt.CudaCompilationException("No capable GPU device has been found. CUDA sources cannot be compiled") from e