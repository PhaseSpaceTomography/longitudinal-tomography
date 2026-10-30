from __future__ import annotations
import re
from os import path
from typing import TYPE_CHECKING

import numpy as np
import yaml

import longitudinal_tomography.utils.tomo_data as tdata

if TYPE_CHECKING:
    from longitudinal_tomography.data.profiles import Profiles
    from longitudinal_tomography.tracking import Machine
    from longitudinal_tomography.utils.tomo_data import Frames

def load_data() -> tuple[Machine, Frames, Profiles]:
    base_dir = path.split(path.realpath(__file__))[0]
    data_path = path.join(base_dir, 'resources')
    dat_path = path.join(data_path, 'INDIVShavingC325.dat')
    data = tdata.TomographyData.from_file(dat_path)
    profiles = data.to_profiles()

    return profiles.machine, data.frames, profiles


def get_machine_args() -> dict:
    test_root = path.split(path.abspath(__file__))[0]
    yml_path = path.join(test_root, 'machine_args.yml')

    with open(yml_path, 'r') as f:
        machine_args = yaml.full_load(f)

    # parse mathematical expressions
    regex = re.compile(r'.+[\*\+\-\/].+')
    for k, v in machine_args.items():
        if isinstance(v, str) and regex.search(v):
            possible_v = eval(v)
            if isinstance(possible_v, int) \
                    or isinstance(possible_v, float):
                machine_args[k] = possible_v

    return machine_args


def load_waterfall():
    base_dir = path.split(path.realpath(__file__))[0]
    data_path = path.join(base_dir, 'resources')

    waterfall = np.load(path.join(
        data_path, 'waterfall_INDIVShavingC325.npy'))
    return waterfall
