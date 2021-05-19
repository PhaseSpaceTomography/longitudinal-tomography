import re
import longitudinal_tomography.utils.tomo_input as tomoin
import typing as t
from copy import deepcopy
from os import path

import numpy as np
import yaml

import tomo.shortcuts as shortcuts
import tomo.utils.tomo_input as tomoin
from tomo.data.profiles import Profiles
from tomo.tomography import Tomography
from tomo.tracking import Machine
from tomo.utils.tomo_input import Frames

_tomo: Tomography = None
_waterfall: np.ndarray = None
_machine: Machine = None
_frames: Frames = None
_profiles: Profiles = None

test_root = path.split(path.abspath(__file__))[0]


def load_data() -> t.Tuple[Machine, Frames, Profiles]:
    global _machine, _frames, _profiles
    if _machine is not None:
        return _machine, _frames, _profiles

    base_dir = path.split(path.realpath(__file__))[0]
    data_path = path.join(base_dir, 'resources')
    dat_path = path.join(data_path, 'INDIVShavingC325.dat')
    raw_params, raw_data = tomoin.get_user_input(dat_path)

    machine, frames = tomoin.txt_input_to_machine(raw_params)
    machine.values_at_turns()
    waterfall = frames.to_waterfall(raw_data)

    profiles = tomoin.raw_data_to_profiles(
        waterfall, machine, frames.rebin, frames.sampling_time)
    profiles.calc_profilecharge()

    _machine = machine
    _frames = frames
    _profiles = profiles

    return machine, frames, profiles


def get_tomography_params() -> t.Tuple[Tomography, Machine]:
    global _tomo
    if _tomo is not None:
        return deepcopy(_tomo), deepcopy(_machine)
    if _machine is None:
        load_data()

    xp, yp = shortcuts.track(_machine, 0)
    tomo = shortcuts.tomogram(_profiles.waterfall, xp, yp, 2)

    _tomo = tomo
    return deepcopy(tomo), deepcopy(_machine)


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
            print(type(possible_v))
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
