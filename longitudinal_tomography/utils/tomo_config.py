import os
import numpy as np
from warnings import warn

class AppConfig:
    _precision = np.float64
    _gpu_enabled = False
    _active_dict = {}

    @classmethod
    def __update_active_dict(cls, new_dict):
        for key in cls._active_dict.keys():
            if key in globals():
                del globals()[key]
        globals().update(new_dict)
        cls._active_dict = new_dict


    @classmethod
    def get_precision(cls):
        return cls._precision

    @classmethod
    def set_single_precision(cls):
        cls._precision = np.float32
        if cls._gpu_enabled:
            cls.load_modules_and_refresh_kernels(True)

    @classmethod
    def set_double_precision(cls):
        cls._precision = np.float64
        if cls._gpu_enabled:
            cls.load_modules_and_refresh_kernels(False)

    @classmethod
    def load_modules_and_refresh_kernels(cls, single_prec_flag=False):
        cls.compile_kernels() # Check if compilation needed
        gpu_dev = GPUDev.get_gpu_dev()
        if single_prec_flag:
            gpu_dev.load_single_precision_modules()
        else:
            gpu_dev.load_double_precision_modules()
        from longitudinal_tomography.python_routines import kick_and_drift_cuda, reconstruct_cuda
        kick_and_drift_cuda.refresh_kernels()
        reconstruct_cuda.refresh_kernels()

    @classmethod
    def is_gpu_enabled(cls):
        return cls._gpu_enabled

    @classmethod
    def use_cpu(cls):
        """
        Use the CPU to perform the calculations.
        The precision of the functions is not set here.
        It will be automatically inferred when calling the libtomo functions.
        """
        from longitudinal_tomography.cpp_routines import libtomo
        cpu_func_dict = {
            'kick_and_drift': libtomo.kick_and_drift,
            'reconstruct': libtomo.reconstruct,
            'make_phase_space': libtomo.make_phase_space,
            'device': 'CPU'
        }

        for fname in dir(np):
            if callable(getattr(np, fname)) and (fname not in cpu_func_dict) \
                    and (fname[0] != '_'):
                cpu_func_dict[fname] = getattr(np, fname)

        cls.__update_active_dict(cpu_func_dict)
        cls._gpu_enabled = False


    @classmethod
    def use_gpu(cls, gpu_id=0):
        """
        Use the GPU device to perform the calculations
        
        Args:
            gpu_id (int, optional): Device id, default = 0
        """
        if gpu_id < 0:
            return

        import cupy as cp

        single_prec = True if cls._precision == np.float32 else False
        GPUDev(single_prec, gpu_id)

        from longitudinal_tomography.python_routines import kick_and_drift_cuda, reconstruct_cuda, data_treatment

        gpu_func_dict = {
            'kick_and_drift': kick_and_drift_cuda.kick_and_drift_cuda,
            'reconstruct': reconstruct_cuda.reconstruct_cuda,
            'make_phase_space': data_treatment.make_phase_space,
            'device': 'GPU'
        }

        for fname in dir(cp):
            if callable(getattr(cp, fname)) and (fname not in gpu_func_dict):
                gpu_func_dict[fname] = getattr(cp, fname)
        cls.__update_active_dict(gpu_func_dict)
        cls._gpu_enabled = True
    
    @classmethod
    def compile_kernels(cls, force=False):
        """
        Compile CUDA kernels if they have not been compiled yet or if forced.

        Args:
            force (bool, optional): If True, forces the compilation of CUDA kernels
                regardless whether they have already been compiled. Defaults to False.
        """
        from longitudinal_tomography import cuda_kernels
        if force or not cuda_kernels.check_compiled_kernels():
            if not force:
                warn("No compiled CUDA kernels found. Compiling CUDA kernels...")
            cuda_kernels.compile_kernels()


class GPUDev:
    __instance = None

    @classmethod
    def get_gpu_dev(cls):
        if cls.__instance is None:
            cls.__instance = GPUDev()
        return cls.__instance

    def __init__(self, single_prec = False, _gpu_id=0):
        if GPUDev.__instance is not None:
            return
        else:
            GPUDev.__instance = self

        import cupy as cp
        self.id = _gpu_id
        self.dev = cp.cuda.Device(self.id)
        self.dev.use()

        self.name = cp.cuda.runtime.getDeviceProperties(self.dev)['name']
        self.attributes = self.dev.attributes
        self.properties = cp.cuda.runtime.getDeviceProperties(self.dev)

        self.func_dict = {}

        # set the default grid and block sizes
        default_blocks = 2 * self.attributes['MultiProcessorCount']
        default_threads = self.attributes['MaxThreadsPerBlock']
        blocks = int(os.environ.get('GPU_BLOCKS', default_blocks))
        threads = int(os.environ.get('GPU_THREADS', default_threads))
        self.grid_size = (blocks, 1, 1)
        self.block_size = (threads, 1, 1)

        self.directory = os.path.dirname(os.path.realpath(__file__)) + "/"

        ## Compile if needed
        AppConfig.compile_kernels()

        if single_prec:
            self.load_single_precision_modules()
        else:
            self.load_double_precision_modules()

    def load_double_precision_modules(self):
        import cupy as cp
        self.kd_mod = cp.RawModule(path=os.path.join(
                    self.directory, f'../cuda_kernels/kick_and_drift_double.cubin'))
        self.rec_mod = cp.RawModule(path=os.path.join(
                            self.directory, f'../cuda_kernels/reconstruct_double.cubin'))

    def load_single_precision_modules(self):
        import cupy as cp
        self.kd_mod = cp.RawModule(path=os.path.join(
                    self.directory, f'../cuda_kernels/kick_and_drift_single.cubin'))
        self.rec_mod = cp.RawModule(path=os.path.join(
                            self.directory, f'../cuda_kernels/reconstruct_single.cubin'))

    def report_attributes(self):
        # Saves into a file all the device attributes
        with open(f'{self.name}-attributes.txt', 'w') as f:
            for k, v in self.attributes.items():
                f.write(f"{k}:{v}\n")

def cast(arr):
    """
    Cast an array (only floats) to a CuPy array if GPU is enabled, otherwise to a NumPy array

    Args:
        arr (numpy.ndarray or cupy.ndarray): The input array to be cast.

    Returns:
        numpy.ndarray or cupy.ndarray: The input array casted on the current device.
    """
    return cast_to_gpu(arr) if AppConfig.is_gpu_enabled() else cast_to_cpu(arr)

def cast_to_gpu(arr):
    """
    Cast an array (only floats) to GPU

    Args:
        arr (numpy.ndarray or cupy.ndarray): The input array to be cast to a CuPy array.

    Returns:
        cupy.ndarray: The input array on GPU.
    """
    import cupy as cp
    arr = cp.array(arr, dtype=AppConfig.get_precision())
    return arr

def cast_to_cpu(arr):
    """
    Cast an array (only floats) to CPU

    Args:
        arr (numpy.ndarray or cupy.ndarray): The input array to be cast to a NumPy memory.

    Returns:
        numpy.ndarray: The input array on CPU.
    """
    if hasattr(arr, 'get'):
        arr = arr.get().astype(AppConfig.get_precision())
    else:
        arr = np.array(arr, dtype=AppConfig.get_precision())
    return arr

AppConfig.use_cpu()