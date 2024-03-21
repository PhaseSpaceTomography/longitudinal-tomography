from .utils import tomo_config as conf

major_version = 3
minor_version = 4
patch_level = 3

dev_version = -1

__version__ = '{}.{}'.format(
    major_version,
    minor_version,
)

if patch_level != 0:
    __version__ += f'.{patch_level}'

if dev_version != -1:
    __version__ += '-dev{}'.format(dev_version)

def use_gpu():
    conf.AppConfig.use_gpu()

def use_cpu():
    conf.AppConfig.use_cpu()

def set_double_precision():
    conf.AppConfig.set_double_precision()

def set_single_precision():
    conf.AppConfig.set_single_precision()