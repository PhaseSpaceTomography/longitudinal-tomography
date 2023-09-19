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
            global gpu_dev
            gpu_dev = GPUDev.__instance

        import cupy as cp
        self.id = _gpu_num
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

        if os.getenv('SINGLE_PREC') is not None:
            single_precision = True if os.getenv('SINGLE_PREC') == 'True' else False
        else:
            single_precision = False

        if os.getenv('ARCH') is not None:
            arch = os.getenv('ARCH')
        else:
            arch = "sm_75"

        if single_precision:
            print("Single precision")
            self.kd_mod = cp.RawModule(path=os.path.join(
                        self.directory, f'../cuda_kernels/kick_and_drift_single_{arch}.cubin'))
            self.rec_mod = cp.RawModule(path=os.path.join(
                        self.directory, f'../cuda_kernels/reconstruct_single_{arch}.cubin'))
        else:
            print("Double precision")
            self.kd_mod = cp.RawModule(path=os.path.join(
                        self.directory, f'../cuda_kernels/kick_and_drift_double_{arch}.cubin'))
            self.rec_mod = cp.RawModule(path=os.path.join(
                            self.directory, f'../cuda_kernels/reconstruct_double_{arch}.cubin'))

    def report_attributes(self):
        # Saves into a file all the device attributes
        with open(f'{self.name}-attributes.txt', 'w') as f:
            for k, v in self.attributes.items():
                f.write(f"{k}:{v}\n")


gpu_dev = None