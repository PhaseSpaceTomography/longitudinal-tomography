# from .tomo_run import run_file
#
# from .tomo_input import Frames, txt_input_to_machine, raw_data_to_profiles
#
from .tomo_output import create_phase_space_image, show
import os

# Class for encapsulating information about the GPU as used in BLonD

class GPUDev:
    __instance = None

    def __init__(self, _gpu_num=0):
        if GPUDev.__instance is not None:
            return
        else:
            GPUDev.__instance = self

        import cupy as cp
        self.id = _gpu_num
        self.dev = cp.cuda.Device(self.id)
        self.dev.use()

        self.name = cp.cuda.runtime.getDeviceProperties(self.dev)['name']
        self.attributes = self.dev.attributes
        self.properties = cp.cuda.runtime.getDeviceProperties(self.dev)

        # set the default grid and block sizes
        default_blocks = 2 * self.attributes['MultiProcessorCount']
        default_threads = self.attributes['MaxThreadsPerBlock']
        blocks = int(os.environ.get('GPU_BLOCKS', default_blocks))
        threads = int(os.environ.get('GPU_THREADS', default_threads))
        self.grid_size = (blocks, 1, 1)
        self.block_size = (threads, 1, 1)

        directory = os.path.dirname(os.path.realpath(__file__)) + "/"

        self.kd_mod = cp.RawModule(path=os.path.join(
                        directory, f'../cuda_kernels/kick_and_drift.cubin'))
        self.rec_mod = cp.RawModule(path=os.path.join(
                        directory, f'../cuda_kernels/reconstruct.cubin'))

    def report_attributes(self):
        # Saves into a file all the device attributes
        with open(f'{self.name}-attributes.txt', 'w') as f:
            for k, v in self.attributes.items():
                f.write(f"{k}:{v}\n")


gpu_dev = GPUDev()