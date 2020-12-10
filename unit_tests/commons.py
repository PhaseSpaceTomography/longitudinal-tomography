import yaml
import numpy as np
from os import path
import re
from tomo.tracking import machine as mch
import tomo.utils.tomo_input as tomoin
import tomo.tracking.tracking as tracking
import tomo.tomography.tomography as tmo
import tomo.tracking.particles as parts
import copy


_tomo = None
_waterfall = None
_machine = None

test_root = path.split(path.abspath(__file__))[0]


# load raw data once and reconstruct, used to reduce post-processing test times
def load_data():
    machine = mch.Machine(**get_machine_args())

    frame_input_args = {
        'framecount':       150,
        'skip_frames':      0,
        'framelength':      1000,
        'dtbin':            machine.dtbin,
        'skip_bins_start':  170,
        'skip_bins_end':    70,
        'rebin':            1
    }

    raw_data = np.genfromtxt(path.join(test_root, 'resources/raw_data_INDIVShavingC325.dat'))
    frames = tomoin.Frames(**frame_input_args)

    waterfall = frames.to_waterfall(raw_data)

    # waterfall = commons.load_waterfall()

    # waterfall = waterfall[:nprofs]
    # waterfall = waterfall[:, 170:-70]  # make cut
    machine.nbins = len(waterfall[0])

    machine.snpt = 2
    rbn = 1
    machine.nprofiles = len(waterfall)

    machine.values_at_turns()

    recprof = 10

    tracker = tracking.Tracking(machine)
    xp, yp = tracker.track(recprof=recprof)
    xp, yp = parts.physical_to_coords(xp, yp, machine,
                                      tracker.particles.xorigin,
                                      tracker.particles.dEbin)
    xp, yp = parts.ready_for_tomography(xp, yp, machine.nbins)

    tomo = tmo.TomographyCpp(waterfall, xp, yp)
    tomo.run(2)

    global _tomo, _waterfall, _machine
    _tomo = tomo
    _waterfall = waterfall
    _machine = machine


def get_tomography_params():
    if _tomo is None:
        load_data()
    return copy.deepcopy(_tomo), copy.deepcopy(_machine)

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
