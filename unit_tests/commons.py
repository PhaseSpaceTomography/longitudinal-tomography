from __future__ import annotations
import re
import typing as t
from os import path

import numpy as np
import yaml

import longitudinal_tomography.utils.tomo_input as tomoin
from longitudinal_tomography.data.profiles import Profiles
from longitudinal_tomography.tracking import Machine
from longitudinal_tomography.utils.tomo_input import Frames

def load_data() -> t.Tuple[Machine, Frames, Profiles]:
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

    return machine, frames, profiles


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
