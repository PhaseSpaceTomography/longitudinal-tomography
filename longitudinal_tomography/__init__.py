major_version = 3
minor_version = 4
patch_level = 1

dev_version = 4

__version__ = '{}.{}'.format(
    major_version,
    minor_version,
)

if patch_level != 0:
    __version__ += f'.{patch_level}'

if dev_version != -1:
    __version__ += '-dev{}'.format(dev_version)
