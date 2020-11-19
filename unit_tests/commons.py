import yaml
from os import path
import re


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
