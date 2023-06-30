# from .tomo_run import run_file
#
# from .tomo_input import Frames, txt_input_to_machine, raw_data_to_profiles
#
from .tomo_output import create_phase_space_image, show
import os

# Class for encapsulating information about the GPU as used in BLonD
from pyprof import timing

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
        else:
            print("Double precision")
            self.kd_mod = cp.RawModule(path=os.path.join(
                        self.directory, f'../cuda_kernels/kick_and_drift_double_{arch}.cubin'))
        self.rec_mod = cp.RawModule(path=os.path.join(
                        self.directory, f'../cuda_kernels/reconstruct_{arch}.cubin'))
        # self.rec_mod = cp.RawModule(path=os.path.join(
        #                 directory, f'../cuda_kernels/reconstruct.cubin'),
        #                 name_expressions=['back_project<int,int>', 'project', 'clip', 'find_difference_profile',\
        #                                   'count_particles_in_bin', 'calculate_reciprocal', 'compensate_particle_amount',\
        #                                     'create_flat_points'])

    # TODO: Discuss if needed, probably going to be removed soon
    def __compile_template_function(self, codepath, name_expression):
        timing.start_timing("compile_backproject")
        import cupy as cp
        code = open(os.path.join(self.directory, f"../cuda_kernels/{codepath}"), "r").read()
        mod = cp.RawModule(code=code, options=('--use_fast_math', '-std=c++17', '-I/usr/local/cuda-11.8/include'),
                            name_expressions=[name_expression], jitify=True)
        kernel_func = mod.get_function(name_expression)
        self.func_dict[name_expression] = kernel_func
        timing.stop_timing()

    # Function to compile template functions based on a path to the source code and the name for the function
    def get_template_function(self, codepath, name_expression):
        if(name_expression not in self.func_dict):
            self.__compile_template_function(codepath, name_expression)

        return self.func_dict.get(name_expression)


    def report_attributes(self):
        # Saves into a file all the device attributes
        with open(f'{self.name}-attributes.txt', 'w') as f:
            for k, v in self.attributes.items():
                f.write(f"{k}:{v}\n")


gpu_dev = GPUDev()