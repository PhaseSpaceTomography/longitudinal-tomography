major_version = 3
minor_version = 0
patch_level = 0

dev_version = 0

__version__ = '{}.{}.{}'.format(
    major_version,
    minor_version,
    patch_level,
)

if dev_version != 0:
    __version__ += '-dev{}'.format(dev_version)
