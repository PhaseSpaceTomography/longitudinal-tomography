import os
import numpy as np

class AppConfig:
    _precision = np.float64
    _gpu_enabled = False
    _active_dict = {}

    @classmethod
    def __update_active_dict(cls, new_dict):
        if not hasattr(cls._active_dict, 'active_dict'):
            cls._active_dict = new_dict
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
            print("SINGLE")
            from . import GPUDev
            gpu_dev = GPUDev.get_gpu_dev()
            gpu_dev.load_single_precision_modules()

    @classmethod
    def set_double_precision(cls):
        cls._precision = np.float64
        if cls._gpu_enabled:
            from . import GPUDev
            gpu_dev = GPUDev.get_gpu_dev()
            gpu_dev.load_single_precision_modules()

    @classmethod
    def use_cpu(cls):
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


    @classmethod
    def use_gpu(cls, gpu_id=0):
        """Use the GPU device to perform the calculations
        
        Args:
            gpu_id (int, optional): Device id, default = 0
        """
        if gpu_id < 0:
            return

        import cupy as cp

        from . import GPUDev
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

AppConfig.use_cpu()