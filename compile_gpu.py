import subprocess
import os

print('Compiling the CUDA sources')

cuda_path = os.getenv('CUDA_PATH', default='')
if cuda_path != '':
    nvcc = cuda_path + '/bin/nvcc'
else:
    nvcc = 'nvcc'

nvccflags = [nvcc, '--cubin', '-O3', '--use_fast_math']

cuda_sources = [
    'longitudinal_tomography/cuda_kernels/kick_and_drift.cu',
    'longitudinal_tomography/cuda_kernels/reconstruct.cu',
]

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
        print('The CUDA source compilation failed.')
except ImportError:
    print("Package CuPy is not installed. CUDA sources cannot be compiled")
except cp.cuda.runtime.CUDARuntimeError:
    print("No capable GPU device has been found. CUDA sources cannot be compiled")