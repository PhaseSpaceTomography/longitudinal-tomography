# from .tomo_run import run_file
#
# from .tomo_input import Frames, txt_input_to_machine, raw_data_to_profiles
#
from .tomo_output import create_phase_space_image, show
import os

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